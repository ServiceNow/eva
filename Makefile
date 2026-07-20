SHELL:=bash

NOW:=$(shell date -u +%Y%m%dt%H%M%Sz)

# ANSI Select Graphic Rendition escape codes
ITALIC:=3
UNDERLINE:=4
# Colors - For example, set the foreground bright blue: $(FG_BRIGHT)$(BLUE)
FG:=3
BG:=4
FG_BRIGHT:=9
BG_BRIGHT:=10
BLACK:=0
RED:=1
GREEN:=2
YELLOW:=3
BLUE:=4
MAGENTA:=5
CYAN:=6
WHITE:=7
STYLE = \033[$(1)m$(2)\033[m

help:  # Show this help message
	@grep -E '^\w.*:.*  # ' $(MAKEFILE_LIST) | awk -F':.*  # ' '{ \
		num_targets = split($$1, targets, " "); \
		printf "%s\n", $$2; \
		for (i = 1; i <= num_targets; i++) printf "  $(call STYLE,$(FG)$(CYAN),make %s)\n", targets[i]; \
		printf "\n"; \
	}'

FORCE:

install:  # Install dependencies
	uv sync --all-extras

merge-public-repo:  # Open a pull request to merge eva's main branch into eva-internal's main-internal branch
	git fetch origin main
	git push --force internal refs/remotes/origin/main:refs/heads/pr/merge-main
	gh pr create --repo ServiceNow/eva-internal --head pr/merge-main --title "Merge public repo" --body "" \
	|| printf '$(call STYLE,$(FG_BRIGHT)$(RED),Unable to create the pull request.\nLooking for an existing pull request:)\n' \
	&& gh pr view --repo ServiceNow/eva-internal pr/merge-main
	gh pr merge --repo ServiceNow/eva-internal --auto --merge pr/merge-main

output/%/toolkit-data-push:  # Push output file(s) to Toolkit
	eai data push --allow-rename snow.core_llm.voice_agent "output/$*:$*"

output/%/toolkit-data-zip-pull: FORCE  # Zip, pull, and unzip output file(s) from Toolkit (this is faster when there are numerous files)
	eai job new \
		--data snow.core_llm.voice_agent:/mnt/voice_agent:rw \
		--image alpine \
		--restartable \
		--wait SUCCEEDED \
		-- sh -c 'cd /mnt/voice_agent && tar --exclude="*.wav" --exclude="*.mp3" -czf "$*.tar.gz" "$*"'
	cd output && eai data pull snow.core_llm.voice_agent "$*.tar.gz" . && tar -xzf "$*.tar.gz" && rm "$*.tar.gz"

output/%/toolkit-data-pull output/%: FORCE  # Pull output file(s) from Toolkit
	eai data pull snow.core_llm.voice_agent "$*" output

apps/traffic.py/download: %/download:  # Download today's GitHub traffic data and push it to Toolkit
	today=$$(date -u +%Y-%m-%d) && \
	{ gh api "repos/ServiceNow/eva/traffic/views"; echo; } > ".github/traffic/$${today}_views.json" && \
	{ gh api "repos/ServiceNow/eva/traffic/clones"; echo; } > ".github/traffic/$${today}_clones.json" && \
	eai data push $(TOOLKIT_DATA) ".github/traffic/$${today}_views.json" ".github/traffic/$${today}_clones.json" --remote-root $(TOOLKIT_JOB_NAME)
	-@$(MAKE) --no-print-directory $*/toolkit-data-rm $*/toolkit-job-url

# Requires bash, not sh, for $'...' quoting, process substitution, and arrays.
apps/traffic.py/toolkit-data-rm:  # Remove outdated GitHub traffic data from Toolkit
	IFS=$$'\n'; \
	files=($$(comm -23 <(eai data content ls snow.core_llm.eva --path eva_github_traffic/.github/traffic --field name | sort) <(ls .github/traffic) | sed 's|^|/tmp/eva/.github/traffic/|')) && \
	if [ "$${#files[@]}" -gt 0 ]; then eai job exec $(TOOLKIT_JOB) -- rm -v -- "$${files[@]}"; fi

apps/analysis.py/run apps/traffic.py/run: %/run:  # Run app locally
	uv run streamlit run $* --server.runOnSave true

TOOLKIT_ACCOUNT=snow.core_llm
TOOLKIT_DATA=snow.core_llm.eva
TOOLKIT_JOB=$(TOOLKIT_ACCOUNT).$(TOOLKIT_JOB_NAME)
TOOLKIT_JOB_TEMPORARY_NAME=$(TOOLKIT_JOB_NAME)__initializing_$(NOW)

apps/analysis.py/%: TOOLKIT_JOB_NAME=eva_results_analysis
apps/traffic.py/%: TOOLKIT_JOB_NAME=eva_github_traffic

apps/%/toolkit-data-push: PATHS=$(DEFAULT_PATHS)
apps/%/toolkit-data-push: DEFAULT_PATHS+= \
	$* \
	pyproject.toml \
	README.md \
	src/eva/ \
	uv.lock \
	website/public/favicon.svg
apps/analysis.py/toolkit-data-push: DEFAULT_PATHS+= \
	.env \
	apps/audio_plots.py \
	internal/
apps/traffic.py/toolkit-data-push: DEFAULT_PATHS+= \
	.github/traffic/
apps/analysis.py/toolkit-data-push apps/traffic.py/toolkit-data-push: %/toolkit-data-push:  # Push app files to Toolkit
	git ls-files $(PATHS) | eai data push $(TOOLKIT_DATA) --files-from - --remote-root $(TOOLKIT_JOB_NAME)

apps/analysis.py/toolkit-job-clear-name apps/traffic.py/toolkit-job-clear-name: %/toolkit-job-clear-name:  # Rename current Toolkit job as deprecated
	eai job set --name $(TOOLKIT_JOB_NAME)__deprecated_on_$(shell date -u +%Y%m%dt%H%M%Sz) $(TOOLKIT_JOB) --format text --format-template $$'Job $(TOOLKIT_JOB) is now {{ .FullName }}\n'

apps/analysis.py/toolkit-job-get apps/traffic.py/toolkit-job-get: %/toolkit-job-get:  # Get Toolkit job details
	eai job get $(TOOLKIT_JOB)

apps/analysis.py/toolkit-job-info apps/traffic.py/toolkit-job-info: %/toolkit-job-info:  # Get Toolkit job info
	eai job info $(TOOLKIT_JOB)

apps/analysis.py/toolkit-job-kill apps/traffic.py/toolkit-job-kill: %/toolkit-job-kill:  # Kill Toolkit job
	eai job kill $(TOOLKIT_JOB) --format text --format-template $$'Job $(TOOLKIT_JOB) is now {{ .State }}\n'

apps/analysis.py/toolkit-job-log apps/traffic.py/toolkit-job-log: %/toolkit-job-log:  # Follow Toolkit job logs, optionally set TAIL=123
	eai job log -f $(TOOLKIT_JOB) $(if $(TAIL),--tail $(TAIL))

apps/analysis.py/toolkit-job-new: TOOLKIT_JOB_EXTRA_COMMAND=set -a && source .env && set +a &&
apps/analysis.py/toolkit-job-new apps/traffic.py/toolkit-job-new: %/toolkit-job-new:  # Deploy app as new Toolkit job, and once started, kill the old job and switch its name to the new job
	eai job new \
		--account $(TOOLKIT_ACCOUNT) \
		--data $(TOOLKIT_DATA)/$(TOOLKIT_JOB_NAME):/tmp/eva:rw \
		--data snow.core_llm.voice_agent:/mnt/voice_agent:rw \
		--env EVA_OUTPUT_DIR=/mnt/voice_agent \
		--env STREAMLIT_BROWSER_SERVER_ADDRESS=$(subst .,-,$(TOOLKIT_JOB)).job.console.elementai.com \
		--env STREAMLIT_BROWSER_SERVER_PORT=80 \
		--env STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll \
		--env STREAMLIT_SERVER_PORT=8080 \
		--env STREAMLIT_SERVER_RUN_ON_SAVE=true \
		--format text \
		--format-template $$'Job {{ .FullName }} submitted\n' \
		--image python:3.11 \
		--mem 8 \
		--name $(TOOLKIT_JOB_TEMPORARY_NAME) \
		--restartable \
		-- bash -c 'cd ~/eva && $(TOOLKIT_JOB_EXTRA_COMMAND) \
			time curl -LsSf https://astral.sh/uv/install.sh | sh && \
			export PATH=$$PATH:~/.local/bin && \
			time uv sync --extra apps && \
			$(STATIC) uv run streamlit run $*'
	@eai job log --follow $(TOOLKIT_ACCOUNT).$(TOOLKIT_JOB_TEMPORARY_NAME) | tee /dev/stderr | { \
		grep --quiet --max-count 1 "You can now view your Streamlit app in your browser." && { \
			pkill -f "eai job log --follow $(TOOLKIT_ACCOUNT).$(TOOLKIT_JOB_TEMPORARY_NAME)" || true; \
		} || { \
			eai job get $(TOOLKIT_ACCOUNT).$(TOOLKIT_JOB_TEMPORARY_NAME) --field stateInfo; \
			echo "Job $(TOOLKIT_ACCOUNT).$(TOOLKIT_JOB_TEMPORARY_NAME) failed to start."; \
			exit 1; \
		}; \
	}
	-@$(MAKE) --no-print-directory $*/toolkit-job-kill $*/toolkit-job-clear-name
	eai job set --name $(TOOLKIT_JOB_NAME) $(TOOLKIT_ACCOUNT).$(TOOLKIT_JOB_TEMPORARY_NAME) --format text --format-template $$'Job $(TOOLKIT_ACCOUNT).$(TOOLKIT_JOB_TEMPORARY_NAME) is now {{ .FullName }}\n'
	-@$(MAKE) --no-print-directory $*/toolkit-job-url

apps/analysis.py/toolkit-job-url apps/traffic.py/toolkit-job-url: %/toolkit-job-url:  # Print Toolkit job URL
	@printf '$(call STYLE,$(UNDERLINE);$(FG_BRIGHT)$(BLUE),https://$(subst .,-,$(TOOLKIT_JOB)).job.console.elementai.com)\n'
