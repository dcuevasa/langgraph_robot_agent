#!/usr/bin/env python3.11 
# -*- coding: utf-8 -*-

# LangChain imports:
from langchain_core.tools import tool

from dummy_task_module import Task_module

tm = Task_module(
    perception=True,
    speech=True,
    manipulation=True,
    navigation=True,
    pytoolkit=True
)

# Perception Tools
@tool
def find_object(object_name: str) -> bool:
    """
    Searches for a specific object in the environment.
    
    Args:
        object_name: Name of the object to search for
    
    Returns:
        True if the object is found, False otherwise
    """
    return tm.find_object(object_name)

@tool
def count_objects(object_name: str) -> int:
    """
    Counts how many instances of an object are present in the environment.
    
    Args:
        object_name: Name of the object to count
    
    Returns:
        Number of objects found
    """
    return tm.count_objects(object_name)

@tool
def search_for_specific_person(posture: str, specific_characteristic: str) -> bool:
    """
    Searches for a person with specific characteristics.
    
    Args:
        posture: The posture of a person. It can be: "pointing", "name", "raised_hand"
        specific_characteristic: Distinctive characteristic of the person
    
    Returns:
        True if the person is found, False otherwise
    """
    return tm.search_for_specific_person(posture, specific_characteristic)

@tool
def find_item_with_characteristic(class_type: str, characteristic: str, furniture: str = "") -> str:
    """
    Searches for an object that has a specific characteristic.
    
    Args:
        class_type: Type of object to search for
        characteristic: Characteristic of the object
        furniture: Piece of furniture to check (optional)
    
    Returns:
        Name of the found object
    """
    return tm.find_item_with_characteristic(class_type, characteristic, furniture)

@tool
def get_person_gesture() -> str:
    """
    Detects the gesture the person in front of the robot is making.
    
    Returns:
        Description of the detected gesture
    """
    return tm.get_person_gesture()

@tool
def get_all_items(furniture: str = "") -> list:
    """
    Gets a list of all objects on top of a piece of furniture.
    
    Args:
        furniture: Piece of furniture to check (optional)
    
    Returns:
        List of object names
    """
    return tm.get_all_items(furniture)

# Speech Tools
@tool
def speak(text: str) -> bool:
    """
    Makes the robot say a text.
    
    Returns:
        True if speech completed successfully
    """
    return tm.talk(text)

@tool
def listen() -> str:
    """
    Activates speech recognition to listen to the user.
    
    Returns:
        Recognized text from speech
    """
    return tm.speech2text_srv()

@tool
def question_and_answer(question: str) -> str:
    """
    Asks a question and gets an answer.
    
    Args:
        question: The question to ask
    
    Returns:
        Answer to the question
    """
    return tm.q_a(question)

@tool
def answer_question(question: str) -> str:
    """
    Generates an answer to a question.
    
    Args:
        question: The question to answer
    
    Returns:
        Answer to the question
    """
    return tm.answer_question(question)

# Navigation Tools
@tool
def go_to_location(location: str) -> bool:
    """
    Navigates to a specific location.
    
    Args:
        location: Name of the location to go to
    
    Returns:
        True if successfully reached the destination
    """
    return tm.go_to_place(location)

@tool
def go_back() -> bool:
    """
    Returns to the previous starting point.
    
    Returns:
        True if successfully reached the starting point
    """
    return tm.go_back()

@tool
def follow_person() -> bool:
    """
    Makes the robot follow a person.
    
    Returns:
        True if following was successful
    """
    return tm.follow_you()

@tool
def stop_robot() -> bool:
    """
    Immediately stops all robot movements.
    
    Returns:
        True if the robot stopped correctly
    """
    return tm.robot_stop_srv()

# Manipulation Tools
@tool
def ask_for_object(object_name: str) -> bool:
    """
    Requests a person to hand over a specific object.
    
    Args:
        object_name: Name of the requested object
    
    Returns:
        True if the request was successful
    """
    return tm.ask_for_object(object_name)

@tool
def give_object(object_name: str) -> bool:
    """
    Gives an object that the robot is holding.
    
    Args:
        object_name: Name of the object to give
    
    Returns:
        True if the handover was successful
    """
    return tm.give_object(object_name)