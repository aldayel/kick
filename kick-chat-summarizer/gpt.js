const OpenAI = require('openai');
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function summarizeMessages(messages) {
  const chatText = messages.map(m => `${m.username}: ${m.message}`).join('\n');
  const prompt = `Summarize the following live chat in 3-5 sentences, focusing on main topics, trends, and overall mood.\nChat:\n${chatText}`;
  const completion = await openai.chat.completions.create({
    model: 'gpt-3.5-turbo',
    messages: [
      { role: 'system', content: 'You are a helpful assistant for live chat summarization.' },
      { role: 'user', content: prompt },
    ],
    temperature: 0.5,
    max_tokens: 300,
  });
  return completion.choices[0].message.content.trim();
}

module.exports = { summarizeMessages };