const axios = require('axios');

const RAPIDAPI_KEY = process.env.KICK_RAPIDAPI_KEY;

async function fetchKickChat(channel) {
  const url = `https://kick.com/api/v2/channels/${channel}/messages`;
  try {
    const response = await axios.get(url, {
      headers: {
        'X-RapidAPI-Key': RAPIDAPI_KEY,
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching Kick chat:', error.message);
    return [];
  }
}

module.exports = { fetchKickChat };