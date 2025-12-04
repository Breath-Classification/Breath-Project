from moviepy.editor import ImageSequenceClip

# lista plików PNG w kolejności
images = [f"Animation/proba_{i}.png" for i in range(0, 2700, 10)]  # dopasuj do swoich plików
clip = ImageSequenceClip(images, fps=10)
clip.write_videofile("output.mp4")