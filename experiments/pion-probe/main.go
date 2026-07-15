// Pion probe for the Telnyx AI-Assistant anonymous-WebRTC path (VSDK-277).
//
// Purpose: aiortc's ICE (aioice) fails to get caller audio through Telnyx's b2bua-rtc
// (the assistant never hears us). The browser works because it uses libwebrtc's ICE.
// Pion has a browser-class ICE stack. This probe drives the exact same anonymous_login +
// Verto invite flow that PR #173's TelnyxWebRTCClient uses, but with Pion instead of
// aiortc, and reports whether the assistant finally hears us.
//
// It:
//   1. dials wss://rtc.telnyx.com, sends anonymous_login (target_type=ai_assistant),
//   2. builds a Pion PeerConnection, adds an opus track from speech.ogg,
//   3. sends telnyx_rtc.invite with the offer SDP,
//   4. applies the answer, streams the audio, and prints ai_conversation events.
//
// "ASSISTANT HEARD US" prints if a role=user item appears -> Pion's media got through.
//
//   go run . -assistant assistant-473093df-...
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
	"github.com/pion/webrtc/v4/pkg/media/oggreader"
)

const (
	rtcHost   = "wss://rtc.telnyx.com"
	userAgent = "EVA-PionProbe/0.1"
)

type rpc struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      interface{}     `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
}

func iceServers() []webrtc.ICEServer {
	return []webrtc.ICEServer{
		{URLs: []string{"stun:stun.telnyx.com:3478"}},
		{URLs: []string{"stun:stun.l.google.com:19302"}},
		{URLs: []string{"turn:turn.telnyx.com:3478?transport=udp"}, Username: "testuser", Credential: "testpassword"},
		{URLs: []string{"turn:turn.telnyx.com:3478?transport=tcp"}, Username: "testuser", Credential: "testpassword"},
		{URLs: []string{"turns:turn2.telnyx.com:443"}, Username: "testuser", Credential: "testpassword"},
	}
}

func main() {
	assistant := flag.String("assistant", "assistant-473093df-d97a-4eae-b484-413bcbe2fda4", "assistant id")
	oggPath := flag.String("audio", "speech.ogg", "opus/ogg file to send as caller audio")
	seconds := flag.Int("seconds", 40, "call duration")
	flag.Parse()

	fmt.Println("[ws] dialing", rtcHost)
	dialer := *websocket.DefaultDialer
	dialer.HandshakeTimeout = 15 * time.Second
	ws, resp, err := dialer.Dial(rtcHost, nil)
	if err != nil {
		code := 0
		if resp != nil {
			code = resp.StatusCode
		}
		fatal(fmt.Sprintf("dial rtc (http %d): %v", code, err))
	}
	fmt.Println("[ws] connected")
	defer ws.Close()

	callID := uuid.NewString()
	heard := make(chan string, 4)
	answerCh := make(chan string, 1)
	sessCh := make(chan string, 1)

	// --- reader: dispatch Verto messages ---
	go func() {
		for {
			_, raw, err := ws.ReadMessage()
			if err != nil {
				return
			}
			var m rpc
			if json.Unmarshal(raw, &m) != nil {
				continue
			}
			switch {
			case m.Result != nil: // reply to our request
				var r struct {
					Sessid string `json:"sessid"`
					SDP    string `json:"sdp"`
				}
				_ = json.Unmarshal(m.Result, &r)
				if r.Sessid != "" {
					select {
					case sessCh <- r.Sessid:
					default:
					}
				}
				if r.SDP != "" {
					select {
					case answerCh <- r.SDP:
					default:
					}
				}
			case m.Method == "telnyx_rtc.answer" || m.Method == "telnyx_rtc.media":
				var p struct {
					SDP string `json:"sdp"`
				}
				_ = json.Unmarshal(m.Params, &p)
				if p.SDP != "" {
					select {
					case answerCh <- p.SDP:
					default:
					}
				}
				replyOK(ws, m.ID, m.Method)
			case m.Method == "ai_conversation":
				var p struct {
					Item struct {
						Role    string `json:"role"`
						Content []struct {
							Text string `json:"text"`
						} `json:"content"`
					} `json:"item"`
				}
				_ = json.Unmarshal(m.Params, &p)
				if p.Item.Role == "user" {
					txt := ""
					if len(p.Item.Content) > 0 {
						txt = p.Item.Content[0].Text
					}
					heard <- txt
				} else if p.Item.Role == "assistant" && len(p.Item.Content) > 0 {
					fmt.Printf("[assistant] %.70s\n", p.Item.Content[0].Text)
				}
				replyOK(ws, m.ID, m.Method)
			case m.ID != nil:
				replyOK(ws, m.ID, m.Method)
			}
		}
	}()

	// --- anonymous_login ---
	send(ws, "anonymous_login", map[string]interface{}{
		"target_type":  "ai_assistant",
		"target_id":    *assistant,
		"userVariables": map[string]interface{}{},
		"reconnection": false,
		"User-Agent":   map[string]string{"sdkVersion": userAgent, "data": "EVA Pion probe"},
	})
	var sessid string
	select {
	case sessid = <-sessCh:
	case <-time.After(10 * time.Second):
		fatal("no sessid from anonymous_login")
	}
	fmt.Println("[verto] session:", sessid)

	// --- Pion peer connection ---
	pc, err := webrtc.NewPeerConnection(webrtc.Configuration{ICEServers: iceServers()})
	must(err, "new peerconnection")
	pc.OnICEConnectionStateChange(func(s webrtc.ICEConnectionState) {
		fmt.Println("[ice]", s.String())
	})

	track, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus}, "audio", "eva")
	must(err, "new track")
	_, err = pc.AddTrack(track)
	must(err, "add track")

	inbound := 0
	pc.OnTrack(func(t *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		for {
			if _, _, err := t.ReadRTP(); err != nil {
				return
			}
			inbound++
		}
	})

	offer, err := pc.CreateOffer(nil)
	must(err, "create offer")
	gather := webrtc.GatheringCompletePromise(pc)
	must(pc.SetLocalDescription(offer), "set local desc")
	<-gather

	// --- telnyx_rtc.invite ---
	send(ws, "telnyx_rtc.invite", map[string]interface{}{
		"sessid": sessid,
		"sdp":    pc.LocalDescription().SDP,
		"dialogParams": map[string]interface{}{
			"callID":             callID,
			"destination_number": "ai_assistant",
			"audio":              true,
		},
		"User-Agent": userAgent,
	})

	var answer string
	select {
	case answer = <-answerCh:
	case <-time.After(15 * time.Second):
		fatal("no answer SDP")
	}
	must(pc.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeAnswer, SDP: answer}), "set remote desc")

	// --- stream the opus file after a short wait for the greeting ---
	go streamOgg(track, *oggPath)

	deadline := time.After(time.Duration(*seconds) * time.Second)
	var transcripts []string
	for {
		select {
		case t := <-heard:
			fmt.Printf("\n*** ASSISTANT HEARD US: %q ***\n\n", t)
			transcripts = append(transcripts, t)
		case <-deadline:
			fmt.Println("\n==== RESULT ====")
			fmt.Println("ice connection state :", pc.ICEConnectionState().String())
			fmt.Println("inbound RTP packets  :", inbound)
			fmt.Printf("assistant heard us   : %v (%d user transcripts)\n", len(transcripts) > 0, len(transcripts))
			return
		}
	}
}

func streamOgg(track *webrtc.TrackLocalStaticSample, path string) {
	time.Sleep(9 * time.Second) // let the greeting finish
	f, err := os.Open(path)
	if err != nil {
		fmt.Println("[audio] open:", err)
		return
	}
	defer f.Close()
	ogg, _, err := oggreader.NewWith(f)
	if err != nil {
		fmt.Println("[audio] oggreader:", err)
		return
	}
	fmt.Println("[audio] streaming caller speech")
	var lastGranule uint64
	for {
		page, hdr, err := ogg.ParseNextPage()
		if err != nil {
			return
		}
		gap := time.Duration((hdr.GranulePosition-lastGranule)*1000/48000) * time.Millisecond
		lastGranule = hdr.GranulePosition
		if gap == 0 {
			gap = 20 * time.Millisecond
		}
		if err := track.WriteSample(media.Sample{Data: page, Duration: gap}); err != nil {
			return
		}
		time.Sleep(gap)
	}
}

func send(ws *websocket.Conn, method string, params interface{}) {
	p, _ := json.Marshal(params)
	msg, _ := json.Marshal(rpc{JSONRPC: "2.0", ID: uuid.NewString(), Method: method, Params: p})
	_ = ws.WriteMessage(websocket.TextMessage, msg)
}

func replyOK(ws *websocket.Conn, id interface{}, method string) {
	if id == nil {
		return
	}
	res, _ := json.Marshal(map[string]string{"method": method})
	msg, _ := json.Marshal(rpc{JSONRPC: "2.0", ID: id, Result: res})
	_ = ws.WriteMessage(websocket.TextMessage, msg)
}

func must(err error, ctx string) {
	if err != nil {
		fatal(ctx + ": " + err.Error())
	}
}
func fatal(s string) {
	fmt.Fprintln(os.Stderr, "FATAL:", s)
	os.Exit(1)
}
