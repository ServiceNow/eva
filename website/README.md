# EVA Website

A New End-to-end Framework for Evaluating Voice Agents (EVA)

Built with React, TypeScript, Vite, and Tailwind CSS.

## Getting Started

### Prerequisites

- Node.js ^20.19 or >=22.12
- npm

### Installation

```bash
cd website
npm install
```

### Development

Run the development server with hot module replacement:

```bash
npm run dev
```

The site will be available at http://localhost:5173

### Production Build

The production site is served from the committed build in [`docs/`](../docs). You normally don't need to run `npm run build` yourself: the `build-website` pre-commit hook runs it for you whenever there are changes in `website/` or [`configs/prompts/judge.yaml`](../configs/prompts/judge.yaml) (the source of the judge prompts shown on the site), and CI rebuilds it on pull requests from this repository.

To commit website changes without rebuilding the website (e.g. if you are missing Node.js), skip the pre-commit hook:

```bash
SKIP=build-website git commit
```

Preview the production build (the current `docs/`) locally:

```bash
npm run preview
```

### Linting

```bash
npm run lint
```

## Project Structure

```
src/
├── components/
│   ├── architecture/    # System architecture diagram
│   ├── conversation/    # Interactive conversation demo
│   ├── hero/            # Landing page hero section
│   ├── layout/          # Navbar, Section wrapper
│   ├── leaderboard/     # Results tables, charts, and visualizations
│   ├── limitations/     # Known limitations section
│   ├── metrics/         # Metrics explorer and judge prompt viewer
│   └── acknowledgements/
├── data/                # Static data for demos and leaderboard
└── styles/              # Theme configuration
```

The site uses hash-based routing with tabs: intro, architecture, metrics, leaderboard-oss, demo, limitations, and acknowledgements.
