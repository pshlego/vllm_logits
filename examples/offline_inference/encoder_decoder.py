# SPDX-License-Identifier: Apache-2.0
"""
Demonstrate prompting of text-to-text
encoder/decoder models, specifically BART
"""

from vllm_logits import LLM, SamplingParams
from vllm_logits.inputs import (
    ExplicitEncoderDecoderPrompt,
    TextPrompt,
    TokensPrompt,
    zip_enc_dec_prompts,
)


def create_prompts(tokenizer):
    # Test prompts
    #
    # This section shows all of the valid ways to prompt an
    # encoder/decoder model.
    #
    # - Helpers for building prompts
    text_prompt_raw = "Hello, my name is"
    text_prompt = TextPrompt(prompt="The president of the United States is")
    tokens_prompt = TokensPrompt(
        prompt_token_ids=tokenizer.encode(prompt="The capital of France is")
    )
    # - Pass a single prompt to encoder/decoder model
    #   (implicitly encoder input prompt);
    #   decoder input prompt is assumed to be None

    single_text_prompt_raw = text_prompt_raw  # Pass a string directly
    single_text_prompt = text_prompt  # Pass a TextPrompt
    single_tokens_prompt = tokens_prompt  # Pass a TokensPrompt

    # ruff: noqa: E501
    # - Pass explicit encoder and decoder input prompts within one data structure.
    #   Encoder and decoder prompts can both independently be text or tokens, with
    #   no requirement that they be the same prompt type. Some example prompt-type
    #   combinations are shown below, note that these are not exhaustive.

    enc_dec_prompt1 = ExplicitEncoderDecoderPrompt(
        # Pass encoder prompt string directly, &
        # pass decoder prompt tokens
        encoder_prompt=single_text_prompt_raw,
        decoder_prompt=single_tokens_prompt,
    )
    enc_dec_prompt2 = ExplicitEncoderDecoderPrompt(
        # Pass TextPrompt to encoder, and
        # pass decoder prompt string directly
        encoder_prompt=single_text_prompt,
        decoder_prompt=single_text_prompt_raw,
    )
    enc_dec_prompt3 = ExplicitEncoderDecoderPrompt(
        # Pass encoder prompt tokens directly, and
        # pass TextPrompt to decoder
        encoder_prompt=single_tokens_prompt,
        decoder_prompt=single_text_prompt,
    )

    # - Finally, here's a useful helper function for zipping encoder and
    #   decoder prompts together into a list of ExplicitEncoderDecoderPrompt
    #   instances
    zipped_prompt_list = zip_enc_dec_prompts(
        ["An encoder prompt", "Another encoder prompt"],
        ["A decoder prompt", "Another decoder prompt"],
    )

    # - Let's put all of the above example prompts together into one list
    #   which we will pass to the encoder/decoder LLM.
    return [
        single_text_prompt_raw,
        single_text_prompt,
        single_tokens_prompt,
        enc_dec_prompt1,
        enc_dec_prompt2,
        enc_dec_prompt3,
    ] + zipped_prompt_list


# Create a sampling params object.
def create_sampling_params():
    return SamplingParams(
        temperature=0,
        top_p=1.0,
        min_tokens=0,
        max_tokens=20,
    )


# Print the outputs.
def print_outputs(outputs):
    print("-" * 50)
    for i, output in enumerate(outputs):
        prompt = output.prompt
        encoder_prompt = output.encoder_prompt
        generated_text = output.outputs[0].text
        print(f"Output {i + 1}:")
        print(
            f"Encoder prompt: {encoder_prompt!r}\n"
            f"Decoder prompt: {prompt!r}\n"
            f"Generated text: {generated_text!r}"
        )
        print("-" * 50)


def main():
    dtype = "float"

    # Create a BART encoder/decoder model instance
    llm = LLM(
        model="facebook/bart-large-cnn",
        dtype=dtype,
    )

    # Get BART tokenizer
    tokenizer = llm.llm_engine.get_tokenizer_group()

    prompts = create_prompts(tokenizer)
    sampling_params = create_sampling_params()

    # Generate output tokens from the prompts. The output is a list of
    # RequestOutput objects that contain the prompt, generated
    # text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    print_outputs(outputs)


if __name__ == "__main__":
    main()
