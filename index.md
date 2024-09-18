---
layout: workshop
root: .  # Is the only page that doesn't follow the pattern /:path/index.html
permalink: index.html  # Is the only page that doesn't follow the pattern /:path/index.html

# Specify the location of the workshop
# TODO: needs documenting a lot better
venue: "Physics Seminar Room"           # name of location use "online" if online
address: "Level 4, Building 46, Highfield Campus, Southampton, SO17 1BJ"
platform-name: None
country: "GB"         # e.g. gb
language: "EN"        # e.g. en
latitude: "50.909698"    # latitude of location (Soton: 50.909698)
longitude: "-1.404351"   # longitude of location (Soton: -1.404351)

# Specify the start and end date of the workshop
humandate: "27th Sept"      # e.g. "6th August"
humantime: "13:00-16:30"      # e.g. "09:00-17:00"
startdate: 2024-09-27        # e.g. YYYY-MM-DD
enddate: 2024-09-27         # e.g. YYYY-MM-DD


# Specify the details of the instructors and helpers
# Can be list, or single string
instructor:
  - "Sam Mangham"
instructor-email:
  - "s.mangham@soton.ac.uk"
helper:
  - "Colin Sauze"
  - "Adam Hill"
  - "NOC Staff"
---
This workshop will be an intro to a variety of common machine learning techniques using SciKit Learn. Please bring along a laptop!

There's more material than we have time for, so we'll adjust pace based on how everyone is doing. There'll be helpers available if people have questions from working through at their own pace.

<!-- this is an html comment -->

{% comment %} This is a comment in Liquid {% endcomment %}

> ## Prerequisites
> A basic understanding of Python. You will need to know how to write a for loop, if statement, use functions, libraries and perform basic arithmetic. 
> Either of the Software Carpentry Python courses cover sufficient background.
{: .prereq}

{% include links.md %}
