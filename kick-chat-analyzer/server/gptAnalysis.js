const OpenAI = require('openai');

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function analyzeMessage(message) {
  const prompt = `Analyze the following chat message. Return a JSON object with these fields: sentiment (positive, negative, or neutral), emotion (e.g., excitement, frustration, appreciation, etc.), toxicity (0-1 scale), and engagement (0-1 scale, how engaging is this message for the chat).\nMessage: "${message}"`;
  const completion = await openai.chat.completions.create({
    model: 'gpt-3.5-turbo',
    messages: [
      { role: 'system', content: 'You are a helpful assistant for chat moderation and analysis.' },
      { role: 'user', content: prompt },
    ],
    temperature: 0.2,
    max_tokens: 150,
  });
  const text = completion.choices[0].message.content;
  try {
    return JSON.parse(text);
  } catch (e) {
    return { sentiment: 'neutral', emotion: 'unknown', toxicity: 0, engagement: 0 };
  }
}

async function generateTalkingPoints(messages) {
  const chatText = messages.map(m => m.message).join('\n');
  const prompt = `Given the following recent chat messages, generate 3 AI talking points or discussion topics that summarize the main trends or themes. Return as a JSON array of strings.\nMessages:\n${chatText}`;
  const completion = await openai.chat.completions.create({
    model: 'gpt-3.5-turbo',
    messages: [
      { role: 'system', content: 'You are a helpful assistant for chat summarization.' },
      { role: 'user', content: prompt },
    ],
    temperature: 0.5,
    max_tokens: 200,
  });
  const text = completion.choices[0].message.content;
  try {
    return JSON.parse(text);
  } catch (e) {
    return [];
  }
}

module.exports = { analyzeMessage, generateTalkingPoints };