

# `awesome-ai-security`[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[![GitHub license](https://img.shields.io/github/license/gmh5225/awesome-ai-security)](https://github.com/gmh5225/awesome-ai-security/blob/main/LICENSE)

A curated list of AI Security materials and resources for Pentesters, Bug Hunters, and Security Researchers.

```
If you find that some links are not working, you can simply replace the username with gmh5225.
Or you can send an issue for me.
```
> Show respect to all the projects below, perfect works of art :saluting_face:

## How to contribute?
- https://github.com/HyunCafe/contribute-practice
- https://docs.github.com/en/get-started/quickstart/contributing-to-projects

## Skills for AI Agents
This repository provides skills that can be used with AI agents and coding assistants such as [Cursor](https://www.cursor.com/), [OpenClaw](https://docs.openclaw.ai/), [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [Codex CLI](https://github.com/openai/codex), and other compatible tools. Install skills to get specialized knowledge about game security topics.

- https://github.com/vercel-labs/skills [The open agent skills tool - npx skills]

**[View on learn-skills.dev](https://learn-skills.dev/skills/gmh5225/awesome-ai-security)**

**Installation:**
```bash
npx skills add https://github.com/gmh5225/awesome-ai-security --skill <skill-name>
```

**Available Skills:**
| Skill | Description |
|-------|-------------|
| `adversarial-machine-learning` | Adversarial machine learning: adversarial examples, data poisoning, model backdoors, and evasion attacks |
| `ai-powered-pentesting` | AI-powered penetration testing tools, red teaming frameworks, and autonomous security agents |
| `llm-attacks-security` | LLM security attacks: prompt injection, jailbreaking, and data extraction |
| `awesome-ai-security-overview` | Overview of this repository and contribution guidelines |
| `ai-security-tooling` | AI security tooling: detectors, analyzers, guardrails, and benchmarks |

**Example:**
```bash
# Install LLM attacks skill
npx skills add https://github.com/gmh5225/awesome-ai-security --skill llm-attacks-security

# Install multiple skills
npx skills add https://github.com/gmh5225/awesome-ai-security --skill adversarial-machine-learning
npx skills add https://github.com/gmh5225/awesome-ai-security --skill ai-powered-pentesting
```



## AI Security Starter Pack

- **CTFs / Practice**
  - https://aivillage.org/ [AI Village @ DEF CON - LLM Jailbreak Challenges]
  - https://doublespeak.chat/#/handbook [Doublespeak - AI Security Challenges]
  - https://github.com/EasyJailbreak/EasyJailbreak [Framework for adversarial jailbreak prompts]
  - https://github.com/microsoft/AI-Red-Teaming-Playground-Labs [Microsoft AI Red Teaming Playground Labs]
  - https://github.com/schwartz1375/genai-security-training [GenAI Red Teaming Training]

- **Blogs / Resources**
  - https://genai.owasp.org/ [OWASP GenAI Security Project]
  - https://llm-stats.com [LLM Leaderboard]
  - https://www.aidaily.win [AI Daily News]
  - https://baoyu.io/blog/how-to-write-good-prompt [How to Write Good Prompts]
  - https://rootissh.in/ [LLM Pentesting Series Blog]

- **Newsletters / Collections**
  - https://mlsecops.com/podcast [MLSecOps Podcast]
  - https://podcasts.apple.com/ph/podcast/the-genai-security-podcast/id1782916580 [GenAI Security Podcast]
  - https://avidml.org/ [AI Vulnerability Database (AVID)]

- **Certifications / Courses**
  - https://cs229.stanford.edu/ [Stanford CS229: Machine Learning]
  - https://course.fast.ai/ [fast.ai Practical Deep Learning]
  - https://www.coursera.org/specializations/deep-learning [Deep Learning Specialization by Andrew Ng]
  - https://huggingface.co/reasoning-course [Build DeepSeek-R1 like Reasoning Model]



## AI/LLM Guide

- **Foundations**
  - https://d2l.ai/ [Dive into Deep Learning - Interactive book with PyTorch/JAX/TensorFlow]
  - http://neuralnetworksanddeeplearning.com/ [Neural Networks and Deep Learning by Michael Nielsen]
  - https://www.deeplearningbook.org/ [Deep Learning by Goodfellow, Bengio, Courville]
  - https://github.com/karminski/one-small-step [AI/LLM Tutorial]
  - https://github.com/datawhalechina/happy-llm [LLM Principles and Practice Tutorial]
  - https://github.com/rasbt/LLMs-from-scratch [Build LLM from Scratch]
  - https://github.com/naklecha/llama3-from-scratch [LLaMA3 from Scratch]
  - https://github.com/ZJU-LLMs/Foundations-of-LLMs [Foundations of LLMs]

- **Awesome Lists**
  - https://github.com/WangRongsheng/awesome-LLM-resourses [Comprehensive LLM Resources]
  - https://github.com/mahseema/awesome-ai-tools [Awesome AI Tools]
  - https://github.com/Shubhamsaboo/awesome-llm-apps [Awesome LLM Apps]
  - https://github.com/punkpeye/awesome-mcp-servers [Awesome MCP Servers]
  - https://github.com/wong2/awesome-mcp-servers [Awesome MCP Servers]
  - https://github.com/deepseek-ai/awesome-deepseek-integration [Awesome DeepSeek Integration]
  - https://github.com/lmmlzn/Awesome-LLMs-Datasets [Awesome LLMs Datasets]



## AI Security & Attacks

### Prompt Injection
- https://www.lakera.ai/blog/guide-to-prompt-injection [Prompt Injection Guide]
- https://genai.owasp.org/llmrisk/llm01-prompt-injection/ [OWASP LLM01:2025 Prompt Injection]
- https://redbotsecurity.com/prompt-injection-attacks-ai-security-2025/ [Prompt Injection Attacks 2025]
- https://github.com/protectai/rebuff [Self-hardening Prompt Injection Detector]
- https://github.com/NVIDIA/garak [NVIDIA LLM Vulnerability Scanner]
- https://github.com/deadbits/vigil-llm [Detects Prompt Injections and Risky Inputs]
- https://github.com/alphasecio/prompt-guard [Prompt Defense for LLM]
- https://github.com/tml-epfl/llm-adaptive-attacks [Adaptive Attacks on LLMs]
- https://github.com/RomiconEZ/llamator [LLM Vulnerability Testing Framework]

### Adversarial Attacks
- https://gradientscience.org/intro_adversarial/ [Introduction to Adversarial Examples]
- https://cset.georgetown.edu/publication/key-concepts-in-ai-safety-robustness-and-adversarial-examples/ [AI Safety and Adversarial Examples]
- https://github.com/Trusted-AI/adversarial-robustness-toolbox [IBM Adversarial Robustness Toolbox]
- https://github.com/QData/TextAttack [Adversarial Attacks on NLP Models]
- https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2025.pdf [NIST Adversarial ML Taxonomy]
- https://llm-vulnerability.github.io/ [ACL 2024 Tutorial: LLM Vulnerabilities]
- https://github.com/tensorflow/cleverhans [CleverHans - ML Vulnerability Benchmark]
- https://github.com/bethgelab/foolbox [Foolbox - Adversarial Examples Toolbox]
- https://github.com/cchio/deep-pwning [Deep-pwning]

### Poisoning & Backdoors
- https://arxiv.org/abs/2009.02276 [Witches' Brew: Industrial Scale Data Poisoning]
- https://arxiv.org/abs/2402.09179 [Instruction Backdoor Attacks on Customized LLMs]
- https://arxiv.org/abs/2510.07192 [Poisoning Attacks Need Only a Few Points]
- https://arxiv.org/abs/1910.03137 [MNTD: Detecting AI Trojans]
- https://owasp.org/www-project-top-10-for-large-language-model-applications/ [OWASP Top 10 for LLM Applications 2025]
- https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers [LLM Harmful Fine-tuning Papers]

### Privacy & Extraction
- https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting [Extracting Training Data from LLMs]
- https://arxiv.org/abs/2309.10544 [Model Leeching: Extraction Attack on LLMs]
- https://arxiv.org/abs/2301.10226 [Watermark for Large Language Models]
- https://arxiv.org/abs/2103.07853 [Membership Inference Attacks Survey]
- https://arxiv.org/abs/2503.19338 [MIAs on Large-Scale Models Survey]
- https://trustllmbenchmark.github.io/TrustLLM-Website/ [TrustLLM Benchmark]
- https://github.com/stratosphereips/awesome-ml-privacy-attacks [Awesome ML Privacy Attacks]
- https://github.com/chawins/llm-sp [LLM Security Papers]
- https://github.com/journey-ad/gemini-watermark-remover [Client-side Gemini AI image watermark remover - Reverse Alpha Blending]

### Model Security
- https://arxiv.org/html/2507.02737v1 [Steganography Capabilities in Frontier LLMs]
- https://jplhughes.github.io/bon-jailbreaking/ [AI Jailbreaking]
- https://huggingface.co/blog/mlabonne/abliteration [Model Abliteration]
- https://github.com/protectai/llm-guard [LLM Guard - Security Tool]
- https://github.com/protectai/modelscan [ModelScan - Scan Models for Unsafe Code]
- https://github.com/fr0gger/nova-framework [Nova Framework - Jailbreak Detection]
- https://github.com/fr0gger/nova_mcp [Nova MCP Server]
- https://github.com/0xAIDR/AIDR-Bastion [GenAI Protection System]
- https://github.com/CAU-ISS-Lab/AIGT-Detection-Evade-Detection [AI-Generated Text Detection & Evasion]



## AI Pentesting & Red Teaming

### AI-Powered Pentesting
- https://github.com/GreyDGL/PentestGPT [GPT-4 Powered Pentesting Agent]
- https://github.com/zakirkun/guardian-cli [AI-Powered Pentesting CLI with Gemini]
- https://github.com/usestrix/strix [AI Security Pentesting]
- https://github.com/aliasrobotics/cai [CAI - Cybersecurity AI Framework]
- https://github.com/promptfoo/promptfoo [AI Agent Pentesting Framework]
- https://github.com/antoninoLorenzo/AI-OPS [AI Assistant for Penetration Testing]
- https://github.com/yz9yt/BugTrace-AI [AI Automated Web Pentesting]
- https://github.com/six2dez/reconftw_ai [ReconFTW with AI Analysis]
- https://github.com/Ed1s0nZ/CyberStrikeAI [AI-Native Security Testing Platform with 100+ Tools]
- https://github.com/vxcontrol/pentagi [PentAGI - Fully autonomous AI agents for penetration testing]
- https://github.com/KeygraphHQ/shannon [Shannon - Autonomous AI pentester, finds and executes real exploits in web apps]

### AI Red Teaming Tools
- https://github.com/Azure/counterfit [Microsoft ML Penetration Testing Tool]
- https://github.com/Azure/PyRIT [Microsoft Red-Teaming Framework for GenAI]
- https://github.com/meta-llama/PurpleLlama [Meta Open-Source LLM Safety Tools]
- https://github.com/NVIDIA/NeMo-Guardrails [NVIDIA Programmable Guardrails]
- https://github.com/NoDataFound/hackGPT [LLM Toolkit for Offensive Security]
- https://github.com/ipa-lab/hackingBuddyGPT [Autonomous Red-Teaming Agent]

### AI Security MCP Tools
- https://github.com/0x4m4/hexstrike-ai [HexStrike AI - 150+ Cybersecurity Tools MCP]
- https://github.com/cyproxio/mcp-for-security [Pentesting MCP]
- https://github.com/johnhalloran321/mcpSafetyScanner [MCP Safety Scanner]
- https://github.com/Karthikathangarasu/pentest-mcp [Pentest MCP]

### AI-Powered C2
- https://github.com/Red-Hex-Consulting/Ankou [AI C2 Framework]

### AI Password Cracking
- https://github.com/d-sec-net/VPK [AI Automated Password Cracking]



## AI Security Tools & Frameworks

### AI Reverse Engineering
- https://github.com/ZeroDaysBroker/GhidraGPT [GPT Integration for Ghidra]
- https://github.com/jtang613/GhidrAssist [LLM Extension for Ghidra]
- https://github.com/0xeb/windbg-copilot [WinDbg Copilot - Agentic Debugging extension]

### AI Vulnerability Detection
- https://github.com/scabench-org/hound [AI Auditor with Adaptive Knowledge Graphs]
- https://semgrep.dev/ [AI-Assisted SAST]
- https://github.com/squirrelscan/squirrelscan [Website audit tool for agent/LLM workflows (security/performance/SEO)]
- https://github.com/rohansx/vgx [Git pre-commit security scanner with LLM integration (detect AI code + vulnerabilities)]
- https://github.com/HikaruEgashira/vulnhuntrs [AI Web Security Audit Tool]
- https://github.com/xvnpw/ai-security-analyzer [AI Security Doc Generator]
- https://github.com/aress31/burpgpt [BurpGPT - AI Vulnerability Scanning]
- https://github.com/haroonawanofficial/AISA-Scanner [AI Security Scanner]

### AI CVE Analysis
- https://github.com/arschlochnop/VulnWatchdog [CVE Monitoring with GPT Analysis]
- https://github.com/suhasgowtham-x/aegis-security-co-pilot [AI CVE Scanner]
- https://github.com/ucsb-mlsec/VulnLLM-R [Specialized Reasoning LLM for Vulnerability]

### AI OSINT
- https://ai.cylect.io/ [AI OSINT]
- https://github.com/apurvsinghgautam/robin/ [AI-Powered Dark Web OSINT Tool]

### AI Security Libraries
- https://secml.readthedocs.io/ [SecML - Secure and Explainable ML Library]
- https://github.com/google/oss-fuzz-gen [AI Code Audit Fuzzing Tool]
- https://github.com/Invicti-Security/brainstorm [AI Fuzzer for Web Applications]

### AI Agent Security
- https://github.com/peg/rampart [Firewall for AI agents - policy engine for OpenClaw, Claude Code, Cursor, Codex]
- https://github.com/cisco-ai-defense/skill-scanner [Security scanner for agent skills - prompt injection, exfiltration, malicious code]
- https://github.com/huifer/skill-security-scan [CLI to scan Claude Skills for security risks before installing]
- https://github.com/avast/sage [Sage - Agent Detection & Response: guards commands, files, web requests for Claude Code, Cursor, OpenClaw]
- https://github.com/botiverse/agent-vault [Keep secrets hidden from AI agents - placeholder I/O layer, encrypted vault]



## AI Agents & Frameworks

### Agent Frameworks
- https://github.com/microsoft/ai-agents-for-beginners [AI Agents for Beginners]
- https://github.com/openai/openai-agents-js [OpenAI Agent JS]
- https://github.com/openai/openai-agents-python [OpenAI Agent Python]
- https://github.com/e2b-dev/awesome-ai-agents [Awesome AI Agents]
- https://github.com/elizaOS/eliza [Autonomous Agents Framework]
- https://github.com/kyegomez/swarms [Enterprise Multi-Agent Orchestration]
- https://github.com/crewAIInc/crewAI [CrewAI - Autonomous AI Agents]
- https://github.com/pydantic/pydantic-ai [Pydantic AI - Agent Framework]
- https://github.com/kortix-ai/suna [Suna - Open Source AI Agent]
- https://github.com/HKUDS/AutoAgent [AutoAgent - Zero-Code LLM Agent]
- https://github.com/VoltAgent/voltagent [VoltAgent - TypeScript AI Agent]
- https://github.com/langchain-ai/langgraph [LangGraph]
- https://github.com/langchain-ai/langchain [LangChain]
- https://github.com/openonion/connectonion [ConnectOnion - AI Agent Framework for Agent Collaboration]
- https://github.com/voltropy/volt [Volt - Coding agent with lossless context management]
- https://github.com/badlogic/pi-mono [Pi - AI agent toolkit: coding agent CLI, LLM API, TUI/web UI, Slack bot, vLLM pods]
- https://github.com/prateekmedia/claude-agent-sdk-pi [Claude Agent SDK as LLM provider for Pi]
- https://github.com/boshu2/agentops [AgentOps - DevOps layer for coding agents: flow, feedback, memory across sessions]

### RAG Frameworks
- https://github.com/infiniflow/ragflow [Best RAG Solution]
- https://github.com/FareedKhan-dev/all-rag-techniques [All RAG Techniques]
- https://github.com/NirDiamant/RAG_Techniques [RAG Techniques Concepts]
- https://github.com/getzep/graphiti [Dynamic RAG]
- https://github.com/lobehub/lobe-chat [Local RAG System]
- https://github.com/HKUDS/MiniRAG [Mini RAG]
- https://github.com/qhjqhj00/MemoRAG [Long Memory RAG]
- https://github.com/microsoft/PIKE-RAG [PIKE-RAG]

### AI Browser Automation
- https://github.com/browser-use/browser-use [Browser-Use - AI Browser Control]
- https://github.com/browser-use/macOS-use [Computer-Use for macOS]
- https://github.com/web-infra-dev/midscene [Browser-Use Alternative]
- https://github.com/browser-use/workflow-use [Browser-Use Workflow Recording]
- https://github.com/microsoft/magentic-ui [Microsoft Browser-Use Alternative]
- https://github.com/lightpanda-io/browser [Lightpanda - Headless Browser for AI]
- https://github.com/jo-inc/camofox-browser [Camofox - Headless browser server for AI agents, anti-detection]

### MCP Servers
- https://mcp.so/ [MCP Collection Website]
- https://github.com/gmh5225/MCP-Chinese-Getting-Started-Guide [MCP Getting Started Guide]
- https://github.com/microsoft/DebugMCP [VSCode extension that exposes a local MCP server for AI-assisted debugging (multi-language)]
- https://github.com/anthropics/knowledge-work-plugins [Claude plugins repo with skills/connectors/slash commands (MCP integration)]
- https://github.com/co-browser/browser-use-mcp-server [Browser-Use MCP]
- https://github.com/langchain-ai/langchain-mcp-adapters [MCP to LangChain Adapter]
- https://github.com/patruff/ollama-mcp-bridge [Ollama MCP Bridge]
- https://github.com/regenrek/deepwiki-mcp [DeepWiki MCP]
- https://github.com/upstash/context7 [Documentation MCP]



## AI Development & Training

### Training Frameworks
- https://github.com/kvcache-ai/ktransformers [LLM Inference Optimization Framework]
- https://github.com/transformerlab/transformerlab-app [Training Studio]
- https://github.com/Lightning-AI/litgpt [Fine-tuning Framework]
- https://github.com/ml-explore/mlx-lm [MLX LLM Fine-tuning]
- https://github.com/arcee-ai/mergekit [Model Merge Tool]
- https://github.com/PrimeIntellect-ai/prime [Distributed AI Training]
- https://docs.unsloth.ai/basics/tutorials-how-to-fine-tune-and-run-llms [Unsloth Fine-tuning]

### Local Models
- https://github.com/mudler/LocalAI [Local Model Loading Tool]
- https://github.com/guinmoon/LLMFarm [LLM on iOS/macOS]
- https://github.com/huggingface/open-r1 [DeepSeek-R1 Open Source Reproduction]
- https://github.com/exo-explore/exo [AI Cluster Model Running]
- https://github.com/CherryHQ/cherry-studio [Local LLM GUI]
- https://github.com/sauravpanda/BrowserAI [Run Local LLMs in Browser]
- https://github.com/signerlabs/Klee [Local Model Chat + RAG]
- https://github.com/dontizi/rlama [Local Ollama + RAG]

### Uncensored Models
- https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard [Uncensored Model Leaderboard]
- https://erichartford.com/uncensored-models [Uncensored Models Training Guide]
- https://huggingface.co/cognitivecomputations/Dolphin3.0-Llama3.1-8B [Dolphin Uncensored Model]

### Prompts & Rules
- https://github.com/NeoVertex1/SuperPrompt [Super Prompt]
- https://github.com/richards199999/Thinking-Claude [Claude Enhancement Prompt]
- https://github.com/LouisShark/chatgpt_system_prompt [ChatGPT System Prompts Collection]
- https://github.com/PatrickJS/awesome-cursorrules [Awesome Cursor Rules]
- https://github.com/anthropics/prompt-eng-interactive-tutorial [Anthropic Prompt Engineering Tutorial]
- https://github.com/langgptai/wonderful-prompts [Wonderful Prompts Collection]

### Claude Code Skills / Plugins
- https://github.com/VoltAgent/awesome-claude-code-subagents [100+ Specialized Claude Code Subagents Collection]
- https://github.com/Dammyjay93/interface-design [Design Engineering for Claude Code - Consistent UI]
- https://github.com/BehiSecc/VibeSec-Skill [Claude skill for secure code and common vulnerability prevention]
- https://github.com/hamelsmu/claude-review-loop [Claude Code plugin: automated code review loop with Codex]
- https://github.com/blader/humanizer [Remove AI Writing Signs from Text]
- https://github.com/op7418/Humanizer-zh [Humanizer Chinese Version]



## AI Applications

### Chat & Assistant
- https://github.com/open-webui/open-webui [ChatGPT Clone]
- https://github.com/ChatGPTNextWeb/NextChat [NextJS Chat]
- https://github.com/vercel/chat [Chat SDK - TypeScript SDK for chat bots on Slack, Teams, Google Chat, Discord]
- https://github.com/vercel/ai-chatbot [Vercel AI Chatbot]
- https://github.com/block/goose [MCP Desktop Agent]
- https://github.com/openclaw/openclaw [Personal AI assistant across platforms and channels]
- https://github.com/HKUDS/nanobot [Ultra-lightweight personal AI assistant (Clawdbot-inspired)]
- https://github.com/zeroclaw-labs/zeroclaw [ZeroClaw - Rust AI assistant, under 5MB RAM, $10 hardware]
- https://github.com/louisho5/picobot [Picobot - Lightweight self-hosted AI bot, single Go binary]

### AI Deep Research
- https://github.com/assafelovic/gpt-researcher [GPT Researcher]
- https://github.com/bytedance/deer-flow [ByteDance Deep Research]
- https://github.com/LearningCircuit/local-deep-research [Local Deep Research]
- https://github.com/u14app/deep-research [Deep Research NextJS]
- https://github.com/zilliztech/deep-searcher [Local Deep Searcher]
- https://github.com/aakashsharan/research-vault [AI Research Assistant with RAG and Structured Extraction]

### AI Search Engines
- https://github.com/rashadphz/farfalle [AI Search Engine]
- https://github.com/miurla/morphic [AI Search Engine]
- https://github.com/zaidmukaddam/scira [AI Search with xAI]
- https://github.com/khoj-ai/khoj [AI Search with Local Models]

### AI Code Analysis
- https://github.com/gmh5225/CodeLens [Code Analysis Tool for LLM]
- https://github.com/mufeedvh/code2prompt [Code to Prompt Tool]
- https://github.com/yamadashy/repomix [GitHub Summarizer for LLM]
- https://github.com/cyclotruc/gitingest [GitHub Summarizer for LLM]
- https://github.com/ahmedkhaleel2004/gitdiagram [GitHub Diagram Generator]
- https://gitingest.com [GitHub Code Merger for LLM]
- https://deepwiki.com [GitHub Project Deep Search]

### AI Web Scraping
- https://github.com/ScrapeGraphAI/Scrapegraph-ai [AI Web Scraping]
- https://github.com/mishushakov/llm-scraper [LLM Web Scraper]
- https://github.com/samber/the-great-gpt-firewall [Anti-AI Web Scraping]

### AI Social Media
- https://github.com/d60/twikit [Twitter Bot Python]
- https://github.com/elizaOS/agent-twitter-client [Twitter Bot JS]
- https://github.com/blorm-network/ZerePy [Twitter AI Agent Python]
- https://github.com/langchain-ai/social-media-agent [Social Media Automation]



## AI Image & Video

### AI Image Generation
- https://github.com/AUTOMATIC1111/stable-diffusion-webui [Stable Diffusion WebUI]
- https://github.com/leejet/stable-diffusion.cpp [Stable Diffusion C++]
- https://github.com/apple/ml-stable-diffusion [Apple Stable Diffusion]
- https://github.com/ant-research/MagicQuill [Intelligent Image Editing]

### AI Video Generation
- https://github.com/bytedance/LatentSync [Digital Human Video]
- https://github.com/HKUDS/AI-Creator [AI Video Creator]

### AI TTS
- https://github.com/SparkAudio/Spark-TTS [Spark TTS]
- https://github.com/rany2/edge-tts [Edge TTS]

### AI Face Recognition
- https://github.com/serengil/deepface [Face Recognition Matching Library]
- https://github.com/s0md3v/roop [AI Face Swap]



## Benchmarks & Standards

- https://robustbench.github.io/ [Adversarial Robustness Benchmark]
- https://jailbreakbench.github.io/ [LLM Jailbreak Benchmark]
- https://crfm.stanford.edu/helm/air-bench/latest/ [Stanford AI Safety Benchmark]
- https://atlas.mitre.org/ [MITRE ATLAS - AI Threat Matrix]
- https://www.nist.gov/itl/ai-risk-management-framework [NIST AI Risk Management Framework]
- https://github.com/vectara/hallucination-leaderboard [Model Hallucination Leaderboard]



## Books

- [Adversarial Machine Learning (Cambridge)](https://www.cambridge.org/core/books/adversarial-machine-learning/C42A9D49CBC626DF7B8E54E72974AA3B) - Building robust ML in adversarial environments
- [Adversarial Learning and Secure AI (Cambridge, 2023)](https://www.cambridge.org/highereducation/books/adversarial-learning-and-secure-ai/79986B5D288511757C2A95D71262E039) - First textbook on adversarial learning
- [Adversarial Robustness for Machine Learning (Elsevier)](https://www.sciencedirect.com/book/9780128240205/adversarial-robustness-for-machine-learning) - Adversarial attack, defense, and verification
- [Machine Learning and Security (O'Reilly)](https://www.oreilly.com/library/view/machine-learning-and/9781491979891/) - ML in cybersecurity
- [Artificial Intelligence: A Modern Approach](https://aima.cs.berkeley.edu/) - AI algorithms background



## Communities & Events

- https://genai.owasp.org/ [OWASP GenAI Security Project]
- https://aivillage.org/ [AI Village @ DEF CON]
- https://avidml.org/ [AI Vulnerability Database]



## Utilities

- https://github.com/JefferyHcool/BiliNote [AI Video Note Generator]
- https://github.com/mediar-ai/screenpipe [AI Screen Monitoring]
- https://github.com/mediar-ai/terminator [AI OCR Recognition Tool]
- https://github.com/gmh5225/git-diff [AI-based Git Commit Message Generator]



## Awesome Lists

- https://github.com/TalEliyahu/Awesome-AI-Security [Governance and Tools Focus]
- https://github.com/ottosulin/awesome-ai-security [Offensive Tools and Labs]
- https://github.com/ElNiak/awesome-ai-cybersecurity [AI in Cybersecurity]
- https://github.com/corca-ai/awesome-llm-security [LLM-specific Security]
- https://github.com/JoranHonig/awesome-web3-ai-security [Web3 AI Security]
- https://github.com/francedot/acu [AI Computer Use Agents]
- https://github.com/hesamsheikh/awesome-openclaw-usecases [OpenClaw Use Cases Collection]
- https://github.com/patchy631/ai-engineering-hub [AI Engineering Hub]
- https://github.com/wgwang/awesome-LLMs-In-China [Chinese LLMs]

