# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.

# Kick.com Live Chat Analyzer

## Local Development

- `npm install`
- `npm run dev` (frontend, http://localhost:5173)
- `cd server && npm start` (backend, http://localhost:5001)

## Render.com Deployment

1. Push this repo to GitHub.
2. Go to [Render.com](https://render.com) and create a new Web Service.
3. Connect your GitHub repo.
4. Set the root directory to `/` (project root).
5. Set the build command: `npm install`
6. Set the start command: `npm start`
7. Add environment variables in Render dashboard:
   - `OPENAI_API_KEY` (your OpenAI key)
   - `KICK_RAPIDAPI_KEY` (your Kick RapidAPI key)
   - `PORT` (set to `10000` or leave blank for Render default)
8. Deploy! Render will build the React app and serve it via Express.

**SQLite database is file-based and will persist as long as the Render disk is not reset. For production, consider a managed DB.**
