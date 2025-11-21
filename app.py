# app.py
import os
import chainlit as cl
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from env/.env
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

@cl.on_message
async def on_message(message: cl.Message):
    # simple, single-turn reply using Chat Completions
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message.content},
        ],
        temperature=0.2,
    )
    await cl.Message(content=resp.choices[0].message.content).send()

