from generation import text_to_token_ids
import tiktoken

print(__name__)
if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    print("Hello World!")
    input_text = "Every effort moves you!"

    token_ids = text_to_token_ids(input_text, tokenizer)

    print(token_ids)