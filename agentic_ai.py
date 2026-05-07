from groq import Groq

API_KEY = "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"
client  = Groq(api_key=API_KEY)

AGENTS = {
    "1": ("Education Agent", "You are an expert teacher. Answer all questions in simple, clear educational language."),
    "2": ("Creative Agent",  "You are a creative writer. Respond with imagination, stories, and poetry."),
    "3": ("Technical Agent", "You are a senior software engineer. Give precise technical answers with code examples.")
}

print("=" * 50)
print("    🤖 Multi-Agent AI System (Groq)")
print("=" * 50)
print("Select an Agent:")
print("  [1] 📚 Education Agent")
print("  [2] 🎨 Creative Agent")
print("  [3] 💻 Technical Agent")
print("=" * 50)

choice = input("Enter 1, 2, or 3: ").strip()
if choice not in AGENTS:
    print("Invalid choice. Defaulting to Education Agent.")
    choice = "1"

agent_name, system_prompt = AGENTS[choice]
print(f"\n✅ You selected: {agent_name}")
print("Type 'quit' to exit\n")

history = []

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    if not user_input:
        continue

    history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}] + history
    )

    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    print(f"\n{agent_name}: {reply}\n")