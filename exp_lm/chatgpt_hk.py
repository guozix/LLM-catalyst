import requests
import json
import os

url = "https://api.openai-hk.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": "hk-xmv49q1000012029eb643fec7cff736180c76930a0f98676"
}

def query_chatgpt(msg, model_id, if_prompt=False, temperature=0.8):
    if if_prompt:
        data = {
            "max_tokens": 1200,
            "model": model_id,
            "temperature": temperature,
            "top_p": 1,
            "presence_penalty": 1,
            "messages": [
                {
                    "role": "system",
                    "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."
                },
                {
                    "role": "user",
                    "content": msg
                }
            ]
        }
    else:
        data = {
            "max_tokens": 1200,
            "model": model_id,
            "temperature": temperature,
            "top_p": 1,
            "presence_penalty": 1,
            "messages": [
                {
                    "role": "user",
                    "content": msg
                }
            ]
        }

    response = requests.post(url, headers=headers, data=json.dumps(data).encode('utf-8') )
    result = response.content.decode("utf-8")
    result = json.loads(result)

    return result

# dummy parse function
def parse_prompt(result):
    ret = []
    for line in result.splitlines():
        line = line.strip()
        if line:
            if '-' == line[0]:
                ret.append(line[1:].strip())
            elif '-' == line[1]:
                ret.append(line[2:].strip())
            elif '.' == line[1]:
                ret.append(line[2:].strip())
            elif '*' == line[0]:
                ret.append(line[1:].strip())
            elif '*' == line[1]:
                ret.append(line[2:].strip())
            else:
                ret.append(line)
    
    for i in range(len(ret)):
        tmp = ret[i]
        if '"' == tmp[0] and '"' == tmp[-1]:
            ret[i] = (tmp[1:-1].strip())
    return ret


def chat_chatgpt(msg, save_chatlog_id=None, *args, **kwargs):
    ret_msg = query_chatgpt(msg, *args, **kwargs)
    ret_content = ret_msg["choices"][0]["message"]["content"]
    print(ret_content)
    
    # write_chat_log
    content_ = {
        "query" : msg,
        "return" : ret_content
    }

    if save_chatlog_id is not None:
        with open(save_chatlog_id , 'w') as f:
            json.dump(content_, f, indent=4)
    return ret_content


def grep_new_prompt_fromlog(chatlog_file):
    with open(chatlog_file , 'r') as f:
        content_ = json.load(f)
    ret_p = parse_prompt(content_['return'])
    print(ret_p)
    return ret_p


def gen_new_prompts(cur_prompts, save_chatlog_id, model_id="gpt-3.5-turbo-1106"):
    msg = """Hi GPT, assume you are a prompt pattern learner.
I have a list of text templates with their corresponding loss values and accuracy. They are used for text classification with pre-trained language model. The templates are arranged in descending order based on their loss value on validation samples, where lower loss indicates better quality.
{}

There are latent patterns that make the template good.
Based on these patterns, write your new template that is different from the old ones and has a loss as low as possible.

Here are some requirements
- Please reply with only the template
- Keep every template under 9 words
- Generate 3 templates that potentially have better image classification performance
"""
    formated_prompts = '\n'.join([f'Templates: {i[0]}\nLoss: {i[1]:.3f}\nAccuracy: {i[2]:.3f}\n' for i in cur_prompts])
    msg_ = msg.format(formated_prompts)
    ret_content = chat_chatgpt(msg_, save_chatlog_id, model_id)
    
    ret_p = parse_prompt(ret_content)
    print(ret_p)

    return ret_p
