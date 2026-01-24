# API Keys Setup Guide

The DataBrain AI Agent requires at least one API key to function. Follow these steps to configure your API keys.

## Quick Setup

1. **Locate the `.env` file** in the project root directory (`/Users/pawankonwar/DataBrain-AI-Agent/.env`)

2. **Open the `.env` file** in a text editor

3. **Add your API keys** (at least one is required):
   ```bash
   # OpenAI API Key
   OPENAI_API_KEY=sk-your-actual-openai-key-here
   
   # DeepSeek API Key
   DEEPSEEK_API_KEY=your-actual-deepseek-key-here
   ```

4. **Save the file**

5. **Restart the server** for changes to take effect

## Getting API Keys

### OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (it starts with `sk-`)
5. Paste it into your `.env` file

### DeepSeek API Key
1. Go to https://platform.deepseek.com/api_keys
2. Sign in or create an account
3. Create a new API key
4. Copy the key
5. Paste it into your `.env` file

## Verification

After adding your keys, restart the server and check the logs:

```bash
tail -50 /tmp/databrain_server.log | grep "API keys"
```

You should see:
```
✓ API keys configured: OpenAI
```
or
```
✓ API keys configured: OpenAI, DeepSeek
```

## Health Check

You can also check the API key status via the health endpoint:

```bash
curl http://localhost:8000/api/health
```

This will return:
```json
{
  "status": "healthy",
  "api_keys_configured": {
    "openai": true,
    "deepseek": false
  },
  "has_at_least_one_key": true
}
```

## Troubleshooting

### Error: "No API keys configured"
- Make sure the `.env` file exists in the project root
- Check that the keys are not empty or placeholder values
- Verify the keys don't have extra spaces or quotes
- Restart the server after making changes

### Error: "API Keys Not Properly Configured"
- Your `.env` file has placeholder values like `your_openai_api_key_here`
- Replace them with your actual API keys

### Error: "LLM configuration error"
- Check that at least one API key is valid
- Verify the keys are correctly formatted (no extra spaces)
- Make sure you've restarted the server after updating `.env`

## Security Notes

- **Never commit your `.env` file to version control** (it's already in `.gitignore`)
- Keep your API keys secret and don't share them
- If a key is compromised, revoke it immediately and generate a new one
- Use environment variables in production instead of `.env` files

## Example `.env` File

```bash
# DataBrain AI Agent - Environment Variables
# IMPORTANT: Add your actual API keys below

# OpenAI API Key (required for OpenAI models)
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# DeepSeek API Key (required for DeepSeek models)
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Note: You need at least ONE of the above API keys configured.
```
