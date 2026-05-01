def create_prompt_text(prompt: str, context: str, tokenizer) -> str:
    enhanced_prompt = f"""<thinking>
Let me solve this coding problem step by step using the provided context:

1. Problem Analysis:
   - Requirement: {prompt}
   - Review relevant context from Stack Overflow

2. Context Analysis:
   {context[:1000]}

3. Solution Approach:
   - Identify the most relevant solution pattern
   - Consider code quality and best practices
   - Evaluate time and space complexity

4. Implementation Strategy:
   - Break down into logical steps
   - Handle edge cases
   - Add error handling and comments

5. Optimization Check:
   - Review for efficiency
   - Check for redundant operations
   - Ensure clean, readable code
</thinking>

<solution>
```python

</solution>

<explanation>
Complexity Analysis:
- Time Complexity: O(n) with justification
- Space Complexity: O(1) with justification

Key Improvements:
- Based on highest-rated Stack Overflow answers
- Optimized for production use
- Handles edge cases
</explanation>"""

    messages = [
        {"role": "system", "content": "You are an expert coding assistant. Use Chain-of-Thought reasoning with structured output blocks (<thinking>, <solution>, <explanation>). Provide optimal, well-commented code following industry best practices."},
        {"role": "user", "content": f"Context from documentation:\n{context}\n\nQuestion: {prompt}\n\n{enhanced_prompt}"}
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)