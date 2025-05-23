# What's an agent?

The very notion of “agency” is hotly debated: what exactly makes something an agent, and what counts as agentic behavior? Opinions also diverge on whether agency should be defined by the outcomes or by the methods used achieve them. As large language models gain direct access to tools and introduce capabilities like deep (re)search, the line between simple automation and genuine agency grows ever harder to draw.

My preferred example of agentic behavior is captured by the ReAct framework [Yao et al., 2022](https://arxiv.org/abs/2210.03629), where a host application with access to "tools" is having a conversation and receiving "orders" from an LLM.

My preferered example of an agentic task (or rather, a task – I think – requires agentic behavior) is something like the [Proust questionnaire](https://en.wikipedia.org/wiki/Proust_Questionnaire), where the interviewee is being asked a set of questions by the interviewer.

To the best of my knowledge, no off-the-shelf LLM can handle this task end-to-end. A model can explain the Proust Questionnaire, list its questions, or even draft the transcript of the interview; but it cannot autonomously run the exchange itself: selecting each question, posing it, collecting responses, and concluding once the entire set is complete.

That gap is easy to close: wrap the model in a simple control loop that, on every iteration, selects the next question, asks it, logs the response, and exits when the list is exhausted. This echoes the *"Agent = (model + tools) in a loop"* that was [stated](https://simonwillison.net/2025/May/22/tools-in-a-loop/) recently at the Anthropic Developer conference.

## A concrete  **(model + tools) in a loop**

Here is a concrete [script](agent_loop_01.py) (< 100 lines of code) you can run, inspired by and adapted from [https://sketch.dev/blog/agent_loop.py](https://sketch.dev/blog/agent_loop.py).

### The interaction
```
% uv run agent_loop_1.py
Reading inline script metadata from `agent_loop_01.py`
Type 'exit' to end the conversation.

Agent:  Hi there! Let's jump right in.

What is your idea of perfect happiness?
You: a book
Agent:  That's a nice simple pleasure! Nothing like getting lost in a good book.

What trait do you most deplore in yourself?
You: procrastination
Agent:  I think many people can relate to that! It can be tough to overcome.

For a final question: What do you consider your greatest achievement?
You: family
Agent:  A meaningful and heartfelt answer. Family is certainly something to be proud of.

Thank you for sharing your thoughts with me! I appreciate your responses.

json
{
  "perfect_happiness": "a book",
  "most_deplored_trait": "procrastination",
  "greatest_achievement": "family"
}

<END>
```

### The loop
This is the loop at its core
```python
 1 def loop(llm):
 2    msg = [{"type": "text", "text": "ready when you are."}]
 3    while True:
 4       output, tool_calls = llm(msg)
 5      print("Agent: ", output)
 6       if output.endswith("<END>"):
 7           raise SystemExit(0)
 8       msg = user_input()
```

Line 2: we inject this prompt (on behalf of the user) to trigger the LLM.

Line 3: this is the loop. We make it infinite but there could be limit based on time, number of rounds, number of tokens, etc.

Line 4: this is where the LLM gets called with the current message. The LLM returns an answer and may ask for some tools to be called.

Line 5: we print the output for the user.

Line 6-7: this is the way the LLM tells that the task is complete.

Line 8: we collect the answer from the user. 

### The prompt
And this is the prompt we use for the LLM
```
You are a helpful AI assistant.
Your job is to ask 3 questions from the Proust Questionnaire to the user.
Don't tell the user the purpose of the conversation.
Don't stop until you have asked all 3 questions and received 3 answers from the user.
Try to be as concise as possible.
Feel free to include some casual chit-chat in between the questions.
It is ok for the user to be off-topic. But bring them back to the task at hand.
When you are done, output the answers in a JSON format, followed by the string "<END>".
```
