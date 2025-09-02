import chainlit as cl

# Predefined FAQ responses
FAQ = {
    "what is your name?": "I am the Panaversity FAQ Bot! 🤖",
    "what can you do?": "I can answer common questions about Panaversity's services and general information. 📚",
    "who created you?": "I was created as part of an assignment project. 📝",
    "how are you?": "I'm just a bot, but I'm always ready to help! ⚡",
    "where can i learn more?": "You can learn more at the Panaversity website. 🌐",
}

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content=(
            "👋 Hello! I'm the Panaversity FAQ Bot.\n\n"
            "You can ask me questions like:\n"
            "- What is your name?\n"
            "- What can you do?\n"
            "- Who created you?\n"
            "- How are you?\n"
            "- Where can I learn more?\n\n"
            "If you ask something else, I'll politely tell you I can only answer predefined questions."
        )
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    user_input = message.content.strip().lower()
    answer = FAQ.get(user_input, "🙏 I'm sorry, I can only answer predefined FAQ questions.")

    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": answer})
    cl.user_session.set("history", history)

    await cl.Message(content=answer).send()