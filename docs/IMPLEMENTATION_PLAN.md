# DataBrain AI Agent - Step-by-Step Implementation Plan

This document outlines the step-by-step implementation of features to avoid dependency hell and test each component independently.

## STEP 1: Chart Generator Tool ✅ (Current Focus)

### Goal
Implement and test the Chart Generator tool independently before integrating with the agent.

### Current Status
- ✅ Chart tool class exists (`databrain_agent/backend/tools/chart_tool.py`)
- ✅ Supports: bar, line, scatter, histogram, box, heatmap
- ✅ Returns base64-encoded images
- ✅ Column validation implemented

### Testing Steps

1. **Test the tool directly:**
   ```bash
   python3 databrain_agent/backend/test_chart_tool.py
   ```

2. **Test via API endpoint:**
   - Start the server: `python3 databrain_agent/backend/main.py`
   - Upload a dataset via `/api/upload-dataset`
   - Test chart generation: `POST /api/test-chart`
   ```json
   {
     "dataset_name": "your_dataset",
     "chart_type": "bar",
     "x_column": "category",
     "title": "Test Chart"
   }
   ```

3. **Verify chart display in frontend:**
   - Charts should display in chat messages
   - Base64 images should render correctly

### Next Steps After Step 1
Once chart generation is verified working:
- ✅ Move to Step 2: RAG Memory

---

## STEP 2: RAG Memory (ChromaDB)

### Goal
Implement RAG memory for storing and retrieving dataset schemas and past analyses.

### Implementation Checklist
- [ ] Verify ChromaDB setup
- [ ] Test schema storage
- [ ] Test context retrieval
- [ ] Test analysis result storage
- [ ] Verify memory persistence

### Testing Steps
1. Test schema storage independently
2. Test context search
3. Verify memory persists across sessions

---

## STEP 3: LangChain Agent Orchestrator

### Goal
Build the full agent orchestrator with tool selection and memory integration.

### Implementation Checklist
- [ ] Agent initialization
- [ ] Tool integration
- [ ] Memory integration
- [ ] Error handling
- [ ] Multi-LLM support

---

## STEP 4: Remaining Tools

### Goal
Add Statistics Calculator and Data Manipulation tools.

### Tools to Implement
- [ ] Statistics Calculator (mean, median, std, correlation)
- [ ] Data Manipulation (filter, sort, group_by, select)

---

## Testing Strategy

Each step should be tested independently:
1. Unit tests for individual components
2. API endpoint tests
3. Integration tests with sample data
4. Frontend display verification

## Current Priority

**Focus: STEP 1 - Chart Generator Tool**

Make sure:
- ✅ Chart tool generates valid base64 images
- ✅ All chart types work (bar, line, scatter, histogram, box, heatmap)
- ✅ Column validation works correctly
- ✅ Charts display in frontend
- ✅ Error handling works

Then proceed to Step 2.
