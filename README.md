## Agent-WebVoyager

### Overview

Agent-WebVoyager is an innovative approach to web navigation and data extraction, capable of performing complex browsing tasks without the need for specific APIs. Mimicking human-like browsing behavior, this agent navigates the web, interacts with pages, and extracts information, all through visual cues and intelligent decision-making processes.

![WebVoyager](path-history/webvoyager.png)

The project showcases the agent's capability to perform a "meta-webscrape" task, such as browsing Twitter to report Elon Musk's most recent tweet, by purely simulating user interactions with the web page. This method stands out by its independence from platform-specific APIs, highlighting a versatile and adaptive web scraping approach.

### Features

- **Human-like Web Navigation:** Employs visual cues and page elements for navigation, making the process similar to how a human would browse.
- **No API Required:** Performs tasks without relying on specific web service APIs, enabling broader applicability across various platforms.
- **Intelligent Decision Making:** Utilizes a set of defined functions to make decisions, interact with web elements, and navigate through pages.
- **Visual Task Documentation:** Generates a visual path history, documenting each step taken during the task execution.

### Installation

To set up Agent-WebVoyager, follow these steps:

1. **Clone the repository:**
   ```
   git clone https://github.com/mrmoxon/Agent-WebVoyager.git
   ```
2. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

### Usage

To run Agent-WebVoyager for a specific task, execute the following command:

```
python agent_voyage.py --task "Browse Twitter and tell me Musk's most recent tweet."
```

The agent will perform up to 25 steps to navigate through the web and accomplish the task.

### Example Task

An example task, "Browse Twitter and tell me Musk's most recent tweet.", demonstrates the agent's ability to perform complex web navigation and information extraction without direct API calls. The agent successfully navigates Twitter, finds Elon Musk's profile, and reports the most recent tweet.

### Task Visual Documentation

The process and steps taken by the agent are documented visually in the path history file: `path-history/agent_path_(twitter).png`. This file illustrates the agent's navigation path, including interactions and key decisions made along the way.

### Agent Path Example

Using objective = "Could you go to Google Trends and compare 'p(doom)' to 'e/acc'?"

![Agent Path Example](path-history/agent_path_(google-trends).png)

*Note: The example image is a representation. Run the agent to generate current task visual documentation.*

### Contributing

Contributions to Agent-WebVoyager are welcome. Please feel free to fork the repository, make changes, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

### License

[MIT](LICENSE)
