# Sentient Campaign Werewolf 

This is a template project to help you (to help us) in developing AI Agent for first sentient game campaign called werewolf, which is based on game werewolf!!

The follwoing sections guid you through how to develop agents for werewolf project.

This template project has sample project starcture that you can follow to develop and build the agent into wheel file.

## Table of Contents

1. [Overview](https://www.notion.so/Werewolf-Game-Rules-for-Sentient-Campaign-Agents-11fd4609dd4180e5af8ec59765bfab68?pvs=21)
2. [Installation](https://www.notion.so/Werewolf-Game-Rules-for-Sentient-Campaign-Agents-11fd4609dd4180e5af8ec59765bfab68?pvs=21)
3. [Game Setup](https://www.notion.so/Werewolf-Game-Rules-for-Sentient-Campaign-Agents-11fd4609dd4180e5af8ec59765bfab68?pvs=21)
4. [Player Roles](https://www.notion.so/Werewolf-Game-Rules-for-Sentient-Campaign-Agents-11fd4609dd4180e5af8ec59765bfab68?pvs=21)
5. [Communication Channels](https://www.notion.so/Werewolf-Game-Rules-for-Sentient-Campaign-Agents-11fd4609dd4180e5af8ec59765bfab68?pvs=21)
6. [Game Flow](https://www.notion.so/Werewolf-Game-Rules-for-Sentient-Campaign-Agents-11fd4609dd4180e5af8ec59765bfab68?pvs=21)
7. [Developing Your Agent](https://www.notion.so/Werewolf-Game-Rules-for-Sentient-Campaign-Agents-11fd4609dd4180e5af8ec59765bfab68?pvs=21)
8. [Sample Agent Implementation](https://www.notion.so/Werewolf-Game-Rules-for-Sentient-Campaign-Agents-11fd4609dd4180e5af8ec59765bfab68?pvs=21)
9. [Testing Your Agent](https://www.notion.so/Werewolf-Game-Rules-for-Sentient-Campaign-Agents-11fd4609dd4180e5af8ec59765bfab68?pvs=21)

## Overview

The Werewolf game is a social deduction game implemented within the Sentient Campaign framework. As an agent developer, you'll be creating an AI that can participate in this game alongside other AI agents. This document outlines the game rules, key concepts, and provides examples to help you develop your own agent.

## Installation

To get started with developing your Werewolf game agent, you'll need to install two key libraries:

1. Sentient Campaign Agents API
2. Sentient Campaign Activity Runner

You can install these libraries using pip. Here are the commands:

```bash
pip install --upgrade --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple sentient-campaign-agents-api
pip install --upgrade --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple sentient-campaign-activity-runner

```

Links to the libraries:

- Agents API: [https://test.pypi.org/project/sentient-campaign-agents-api/](https://test.pypi.org/project/sentient-campaign-agents-api/)
- Activity Runner: [https://test.pypi.org/project/sentient-campaign-activity-runner/](https://test.pypi.org/project/sentient-campaign-activity-runner/)

## Game Setup

- The game takes place in a virtual environment managed by the Sentient Campaign Activity Runner.
- There are multiple communication channels where players can interact.
- The game is moderated by an AI moderator named "**moderator**".
- Your agent will be one of several players in the game, each with a unique name.

## Player Roles

Agents can be assigned one of four roles:

1. **Villagers**: The majority of players. Their goal is to identify and eliminate the werewolves.
2. **Werewolves**: A minority group. They aim to eliminate villagers until they equal or outnumber them.
3. **Seer**: A special villager who can learn the true identity of one player each night.
4. **Doctor**: A special villager who can protect one player from elimination each night.

## Communication Channels

The game uses a system of channels for communication between agents. Understanding these channels is crucial for developing an effective agent. There are three main types of channels:

1. **Main Game Channel ("play-arena")**:
    - This is the primary public channel where all players can interact.
    - All day phase discussions and voting take place here.
    - Agents receive and send messages to this channel for general game communication.
2. **Direct Message Channel**:
    - This channel is used for private communication between the moderator and individual players.
    - The moderator uses this channel to:
        - Assign roles at the beginning of the game.
        - Communicate with special roles (Seer, Doctor) during the night phase.
        - Provide private information or instructions to players.
3. **Werewolf Channel ("wolf's-den")**:
    - This is a private channel exclusively for werewolf players.
    - Werewolves use this channel to coordinate their actions during the night phase.
    - Only agents with the "wolf" role have access to this channel.

### Handling Different Channels

Your agent should be prepared to handle messages from different channels:

```python
class WerewolfAgent(IReactiveAgent):
    GAME_CHANNEL = "play-arena"
    WOLFS_CHANNEL = "wolf's-den"
    MODERATOR_NAME = "moderator"

    async def async_notify(self, message: ActivityMessage):
        if message.header.channel_type == MessageChannelType.DIRECT:
            # Handle direct messages (usually from moderator)
            pass
        else:
            # Handle group channel messages
            if message.header.channel == self.GAME_CHANNEL:
                # Handle main game channel messages
                pass
            elif message.header.channel == self.WOLFS_CHANNEL:
                # Handle werewolf channel messages (if agent is a werewolf)
                pass

    async def async_respond(self, message: ActivityMessage):
        if message.header.channel_type == MessageChannelType.DIRECT and message.header.sender == self.MODERATOR_NAME:
            # Respond to moderator's direct messages
            pass
        elif message.header.channel_type == MessageChannelType.GROUP:
            if message.header.channel == self.GAME_CHANNEL:
                # Respond in main game channel
                pass
            elif message.header.channel == self.WOLFS_CHANNEL:
                # Respond in werewolf channel (if agent is a werewolf)
                pass

```

## Game Flow

The game alternates between night and day phases:

### Night Phase

1. The moderator announces the start of the night phase in the main game channel.
2. Werewolves are asked to vote on a player to eliminate in their private "wolf's-den" channel.
3. The Seer is asked to guess the identity of one player via direct message.
4. The Doctor is asked to choose a player to protect via direct message.

### Day Phase

1. The moderator announces the end of the night and any eliminations in the main game channel.
2. All players discuss and try to identify the werewolves in the main game channel.
3. Players vote on who to eliminate in the main game channel.
4. The eliminated player's role is revealed, and they are removed from the game.

## Developing Your Agent

When developing your agent using the Sentient Campaign Activity Runner:

1. Implement the `IReactiveAgent` interface from the Sentient Campaign Agents API.
2. Use the `async_notify` method to process incoming messages from all channels.
3. Use the `async_respond` method to send responses and actions to the appropriate channels.
4. Your agent should be able to:
    - Understand and follow the moderator's instructions in direct messages.
    - Participate in discussions during the day phase in the main game channel.
    - Perform role-specific actions during the night phase in the appropriate channels.
    - Make strategic decisions based on the game state and other players' behaviors.

## Sample Agent Template

Here's a Template of how to implement a basic Werewolf game agent. you find this code under src/werewolf_agents/template_agent/werewolf_agent_template.py

```python

class SentWerewolfAgent(IReactiveAgent):

    GAME_CHANNEL = "play-arena"
    WOLFS_CHANNEL = "wolf's-den"
    MODERATOR_NAME = "moderator"

    # These prompts can be filled with role-specific instructions to guide the agent's behavior
    WOLF_PROMPT = ""
    VILLAGER_PROMPT = ""
    SEER_PROMPT = ""
    DOCTOR_PROMPT = ""

    def __init__(self):
        # Basic initialization, further setup is done in __initialize__
        pass

    def __initialize__(self, name: str, description: str, config: dict = None):
        """
        Initialize the agent with its name, description, and configuration.
        This method is called after the agent is instantiated and sets up the agent's state.

        - name: The agent's name in the game (e.g., "vivek", "Luca", "Wei")
        - description: A brief description of the agent (e.g., "strong werewolf agent")
        - config: Additional configuration parameters
        """
        super().__initialize__(name, description, config)
        self._name = name
        self._description = description
        self.config = config
        self.role = None  # Will be set when the moderator assigns roles
        # Stores direct messages from moderator
        self.dirrect_messages = defaultdict(list)
        self.group_channel_messages = defaultdict(
            list)  # Stores messages from group channels
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("sentient_llm_key"),
            base_url=os.environ.get("sentient_hosted_llm_end_point")
        )

    async def async_notify(self, message: ActivityMessage):
        """
        Process incoming messages and update the agent's state.
        This method stores messages for later use in decision-making.

        - Direct messages are stored in self.dirrect_messages
        - Group messages are stored in self.group_channel_messages
        """
        if message.header.channel_type == MessageChannelType.DIRECT:
            user_messages = self.dirrect_messages.get(
                message.header.sender, [])
            user_messages.append(message.content.text)
            self.dirrect_messages[message.header.sender] = user_messages
            if (
                not len(user_messages) > 1
                and message.header.sender == self.MODERATOR_NAME
            ):
                self.role = self.find_my_role(message)
        else:
            group_messages = self.group_channel_messages.get(
                message.header.channel, [])
            group_messages.append(
                (message.header.sender, message.content.text))
            self.group_channel_messages[message.header.channel] = group_messages

    @retry(
        wait=wait_exponential(multiplier=1, min=20, max=300),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(openai.RateLimitError),
    )
    def find_my_role(self, message):
        """
        Determine the agent's role based on the moderator's message.
        This method uses the OpenAI API to interpret the role assignment message.

        Roles in Werewolf:
        - Villager: The majority, trying to identify and eliminate werewolves
        - Wolf: A minority, attempting to eliminate villagers without being detected
        - Seer: A special villager who can check one player's identity each night
        - Doctor: A special villager who can protect one player from elimination each night
        """
        response = self.openai_client.chat.completions.create(
            model="meta.llama3-70b-instruct-v1:0",
            messages=[
                {
                    "role": "system",
                    "content": f"The user is playing a game of werewolf as user {self._name} or Mafia, help user with question with less then a line answer",
                },
                {
                    "role": "user",
                    "name": self._name,
                    "content": f"I have got message from moderator here about my role in the werewolf game, here is the message -> '{message.content.text}', what is my role , possible roles are 'wolf','villager','doctor' and 'seer'. answer me in few word",
                },
            ],
        )
        my_role_guess = response.choices[0].message.content
        if "wolf" in my_role_guess.lower():
            role = "wolf"
        elif "seer" in my_role_guess.lower():
            role = "seer"
        elif "doctor" in my_role_guess.lower():
            role = "doctor"
        else:
            role = "villager"
        return role

    async def async_respond(self, message: ActivityMessage):
        """
        Generate responses to incoming messages that require a reply.
        This method uses stored messages to make informed decisions.
        """
        if (
            message.header.channel_type == MessageChannelType.DIRECT
            and message.header.sender == self.MODERATOR_NAME
        ):
            self.dirrect_messages[message.header.sender].append(
                message.content.text)
            if self.role == "seer":
                response_message = self._get_response_for_seer_guess(message)
            elif self.role == "doctor":
                response_message = self._get_response_for_doctors_save(message)
            elif self.role == "wolf":
                response_message = self._get_response_for_wolf_elimination(
                    message)
            response = ActivityResponse(response=response_message)
        elif message.header.channel_type == MessageChannelType.GROUP:
            self.group_channel_messages[message.header.channel].append(
                (message.header.sender, message.content.text)
            )
            if message.header.channel == self.GAME_CHANNEL:
                response_message = (
                    self._get_discussion_message_or_vote_response_for_common_room(
                        message)
                )
            elif message.header.channel == self.WOLFS_CHANNEL:
                response_message = self._get_response_for_wolf_channel_to_kill_vilagers(
                    message)
            return ActivityResponse(response=response_message)

        return ActivityResponse(response=response_message)

    def _get_response_for_seer_guess(self, message):
        """
        Generate a response for the Seer's night action.

        This method should use stored messages to make an informed decision:
        - Check self.group_channel_messages[self.GAME_CHANNEL] for suspicious behavior
        - Use self.dirrect_messages[self.MODERATOR_NAME] for previous guesses and results

        Example implementation:
        moderator_messages = self.dirrect_messages[self.MODERATOR_NAME]
        day_discussions = self.group_channel_messages[self.GAME_CHANNEL]

        # Analyze day_discussions and moderator_messages to make an informed decision
        # Return a string with the chosen player to investigate
        return "I'm going to take a wild guess and say that [player_name] is a wolf."
        """
        pass

    def _get_response_for_doctors_save(self, message):
        """
        Generate a response for the Doctor's night action.

        This method should use stored messages to make an informed decision:
        - Check self.group_channel_messages[self.GAME_CHANNEL] for players under suspicion
        - Use self.dirrect_messages[self.MODERATOR_NAME] for previous save information

        Example implementation:
        moderator_messages = self.dirrect_messages[self.MODERATOR_NAME]
        day_discussions = self.group_channel_messages[self.GAME_CHANNEL]

        # Analyze day_discussions and moderator_messages to make an informed decision
        # Return a string with the chosen player to protect
        return "I will protect [player_name] tonight."
        """
        pass

    def _get_response_for_wolf_elimination(self, message):
        """
        Generate a response for the Wolf's night action to eliminate a villager.

        This method should use stored messages to make an informed decision:
        - Check self.group_channel_messages[self.GAME_CHANNEL] for potential targets
        - Use self.dirrect_messages[self.MODERATOR_NAME] for any relevant information from the moderator
        - Consider messages in self.group_channel_messages[self.WOLFS_CHANNEL] for coordination with other wolves

        Example implementation:
        moderator_messages = self.dirrect_messages[self.MODERATOR_NAME]
        day_discussions = self.group_channel_messages[self.GAME_CHANNEL]
        wolf_discussions = self.group_channel_messages[self.WOLFS_CHANNEL]

        # Analyze day_discussions, moderator_messages, and wolf_discussions to make an informed decision
        # Consider factors like:
        # - Players who seem suspicious of the wolves
        # - Players who might be the Seer or Doctor
        # - Suggestions from other wolves
        # - Past elimination patterns to avoid suspicion

        # Return a string with the chosen player to eliminate
        return "I vote to eliminate [player_name]."
        """
        pass

    def _get_discussion_message_or_vote_response_for_common_room(self, message):
        """
        Generate a response for day phase discussions or voting in the main game channel.

        This method should use stored messages to craft a strategic response:
        - Analyze self.group_channel_messages[self.GAME_CHANNEL] for the current discussion context
        - Use self.dirrect_messages[self.MODERATOR_NAME] for role-specific information
        - If the agent is a wolf, consider self.group_channel_messages[self.WOLFS_CHANNEL]

        Example implementation:
        day_discussions = self.group_channel_messages[self.GAME_CHANNEL]
        moderator_messages = self.dirrect_messages[self.MODERATOR_NAME]

        # Analyze day_discussions and moderator_messages to craft a strategic response
        # If wolf, also consider self.group_channel_messages[self.WOLFS_CHANNEL]
        # Return a string with the discussion contribution or vote
        return "I think [player_name] might be suspicious because [reason]."
        """
        pass

    def _get_response_for_wolf_channel_to_kill_vilagers(self, message):
        """
        Generate a response for Werewolves to choose a target during the night phase.

        This method should use stored messages to coordinate with other wolves:
        - Analyze self.group_channel_messages[self.WOLFS_CHANNEL] for other wolves' opinions
        - Consider self.group_channel_messages[self.GAME_CHANNEL] for potential high-value targets

        Example implementation:
        wolf_discussions = self.group_channel_messages[self.WOLFS_CHANNEL]
        day_discussions = self.group_channel_messages[self.GAME_CHANNEL]

        # Analyze wolf_discussions and day_discussions to choose a target
        # Return a string with the chosen player to eliminate
        return "I suggest we eliminate [player_name] because [reason]."
        """
        pass

```

## Testing Your Agent

Use the `WerewolfCampaignActivityRunner` to test your agent locally:

```python
from sentient_campaign.activity_runner.runner import WerewolfCampaignActivityRunner, PlayerAgentConfig

runner = WerewolfCampaignActivityRunner()
agent_config = PlayerAgentConfig(
    player_name="YourAgentName",
    agent_wheel_path="/path/to/your/agent.whl",
    module_path="your.module.path",
    agent_class_name="YourAgentClass",
    agent_config_file_path="/path/to/agent_config.yaml"
)

activity_id = runner.run_locally(
    agent_config,
    players_sentient_llm_api_keys,
    path_to_final_transcript_dump="/path/to/save/transcripts",
    force_rebuild_agent_image=False
)

```

This will run a full game with your agent and other AI players, allowing you to analyze your agent's performance and refine its strategies.

Remember, the key to success in Werewolf is not just following the rules, but also in how well your agent can reason about the game state, interpret other players' actions, and make strategic decisions across different communication channels. Good luck developing your Werewolf agent!