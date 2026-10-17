from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates # UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialize our fastapi app

app = FastAPI (title="Text Summerizer APP", description="Text Summarization using T5", version="1.0")

# model and tokenizer

model = T5ForConditionalGeneration.from_pretrained("usamasharif143/text-summarizer-model")
tokenizer = T5Tokenizer.from_pretrained("usamasharif143/text-summarizer-model")


# model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
# tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# Device

if torch.cuda.is_available():
  device = torch.device("cuda")
elif torch.backends.mps.is_available():
  device = torch.device("mps")
else:
  device= torch.device("cpu")
print("device ", device)

model.to(device)

# templating
templete = Jinja2Templates(directory=".")

# input schema(format) for dialogue  ==> string

class DialogueInput(BaseModel):
  dialogue: str

# cleaning data


def clean_data(text):
   text = re.sub(r"\n\r", " ", text) # new lines
   text = re.sub(r"<.*?>", " ", text) # HTML tags
   text = re.sub(r"\s+", " ", text)  # spaces
   text = text.strip().lower()
   return text


def summarize_dialogue(dialogue: str) ->str:
  dialogue = clean_data(dialogue)  # clean


  # tokenize
  inputs = tokenizer(
      dialogue,
      max_length=512,
      padding="max_length",
      truncation=True,
      return_tensors="pt"  # we use pt(pytorch tensor) if we use tensor flow then we use "tf" so model in huggging face are by default pytorch tensor(pt) so that we use "pt"
      )

  # generate the summary  ==> token ids

  target = model.generate(
      input_ids = inputs["input_ids"],
      attention_mask = inputs["attention_mask"],
      max_length=180,
      min_length=50,
      length_penalty=2.0,
      num_beams=4,   # num of possible sequence
      early_stopping=True   # using to stop the num_beams when EOS .. like jub beam 4 tk hu jaye gy then ye auto stop hu jaye ga..
  )

  # token ids convert to summary ==> decording

  summary = tokenizer.decode(target[0], skip_special_tokens=True) # EOS

  return summary


# API EndPoints


@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templete.TemplateResponse(request, "Index.html", {})

