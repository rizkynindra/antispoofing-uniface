---
title: Face Compare
emoji: 👤
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
short_description: Upload two photos and compare the faces.
python_version: "3.12"
startup_duration_timeout: 30m
---

Upload a reference photo and a second photo. The demo returns whether they are the same person. The spoof check is optional and only looks at the second photo.
