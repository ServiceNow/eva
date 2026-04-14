NOW:=$(shell date -u +%Y%m%dt%H%M%Sz)

PRINT_BLUE=printf '\033[4;94m%s\033[m\n'

apps/analysis.py/toolkit-%: TOOLKIT_ACCOUNT=snow.core_llm
apps/analysis.py/toolkit-%: TOOLKIT_DATA=snow.core_llm.eva
apps/analysis.py/toolkit-%: TOOLKIT_JOB_NAME=eva_results_analysis
apps/analysis.py/toolkit-%: TOOLKIT_JOB=$(TOOLKIT_ACCOUNT).$(TOOLKIT_JOB_NAME)
apps/analysis.py/toolkit-%: TOOLKIT_JOB_TEMPORARY_NAME=$(TOOLKIT_JOB_NAME)__initializing_$(NOW)

apps/analysis.py/toolkit-data-push: %/toolkit-data-push:
	for path in \
		internal/ \
		src/eva/ \
		apps/analysis.py \
		pyproject.toml \
		README.md \
		uv.lock \
		website/public/favicon.svg \
	; do eai data push $(TOOLKIT_DATA) $${path}:$(TOOLKIT_JOB_NAME)/$${path}; done

apps/analysis.py/toolkit-job-clear-name:
	eai job set --name $(TOOLKIT_JOB_NAME)__deprecated_on_$(shell date -u +%Y%m%dt%H%M%Sz) $(TOOLKIT_JOB) --format text --format-template $$'$(TOOLKIT_JOB) is now {{ .FullName }}\n'

apps/analysis.py/toolkit-job-get:
	eai job get $(TOOLKIT_JOB)

apps/analysis.py/toolkit-job-info:
	eai job info $(TOOLKIT_JOB)

apps/analysis.py/toolkit-job-kill:
	eai job kill $(TOOLKIT_JOB) --format text --format-template $$'$(TOOLKIT_JOB) is now {{ .State }}\n'

apps/analysis.py/toolkit-job-log:
	eai job log -f $(TOOLKIT_JOB) $(if $(TAIL),--tail $(TAIL))

apps/analysis.py/toolkit-job-new: %/toolkit-job-new:
	eai job new \
		--account $(TOOLKIT_ACCOUNT) \
		--data $(TOOLKIT_DATA)/$(TOOLKIT_JOB_NAME):/tmp/eva:rw \
		--data snow.core_llm.voice_agent:/app/output:rw \
		--env EVA_OUTPUT_DIR=/app/output \
		--env STREAMLIT_BROWSER_SERVER_ADDRESS=$(subst .,-,$(TOOLKIT_JOB)).job.console.elementai.com \
		--env STREAMLIT_BROWSER_SERVER_PORT=80 \
		--env STREAMLIT_CLIENT_TOOLBAR_MODE=developer \
		--env STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll \
		--env STREAMLIT_SERVER_PORT=8080 \
		--env STREAMLIT_SERVER_RUN_ON_SAVE=true \
		--format text \
		--format-template $$'{{ .FullName }} submitted\n' \
		--image python:3.11 \
		--mem 8 \
		--name $(TOOLKIT_JOB_TEMPORARY_NAME) \
		--restartable \
		-- bash -c 'cd ~/eva && \
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
	-@$(MAKE) --no-print-directory $*/toolkit-job-kill
	-@$(MAKE) --no-print-directory $*/toolkit-job-clear-name
	eai job set --name $(TOOLKIT_JOB_NAME) $(TOOLKIT_ACCOUNT).$(TOOLKIT_JOB_TEMPORARY_NAME) --format text --format-template $$'$(TOOLKIT_ACCOUNT).$(TOOLKIT_JOB_TEMPORARY_NAME) is now {{ .FullName }}\n'
	-@$(MAKE) --no-print-directory $*/toolkit-job-url

apps/analysis.py/toolkit-job-url:
	@$(PRINT_BLUE) 'https://$(subst .,-,$(TOOLKIT_JOB)).job.console.elementai.com'
