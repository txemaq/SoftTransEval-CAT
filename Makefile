.PHONY: eval-gemma3 eval-gpt-oss eval-mistral eval-qwen3 all-cloud

all-local: eval-gemma3 eval-gpt-oss eval-mistral eval-qwen3

gemma_prompts := 1 2 2_1 3 3_1 4 5
eval-gemma3:
	@for p in $(gemma_prompts); do \
		python evaluator/evaluator.py --prompt_version $$p; \
		python evaluator/json_to_md.py $$(ls -t ./output/*.json | head -n 1); \
	done

gpt-oss_prompts := 1
eval-gpt-oss:
	@for p in $(gpt-oss_prompts); do \
		python evaluator/evaluator.py --model gpt-oss --prompt_version $$p; \
		python evaluator/json_to_md.py $$(ls -t ./output/*.json | head -n 1); \
	done

mistral_prompts := 1
eval-mistral:
	@for p in $(mistral_prompts); do \
		python evaluator/evaluator.py --model mistral --prompt_version $$p; \
		python evaluator/json_to_md.py $$(ls -t ./output/*.json | head -n 1); \
	done

qwen3_prompts := 1
eval-qwen3:
	@for p in $(qwen3_prompts); do \
		python evaluator/evaluator.py --model qwen3 --prompt_version $$p; \
		python evaluator/json_to_md.py $$(ls -t ./output/*.json | head -n 1); \
	done

models := "gpt-5" "gpt-5-mini" "gemini-2.5-flash" "gemini-2.5-pro"

all-cloud:
	@for model in $(models); do \
		python evaluator/evaluator.py --model $$model; \
	done
	python evaluator/json_to_md.py "$$(ls -1t ./output/*.json | head -n1)"

download-models:
	mkdir -p models
	wget -P models "https://huggingface.co/bartowski/google_gemma-3-27b-it-GGUF/resolve/main/google_gemma-3-27b-it-Q4_0.gguf"
	

