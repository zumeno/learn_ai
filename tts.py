from gtts import gTTS

def text_to_speech(text, output_file, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save(output_file)

text_to_speech("This is a test on GTTS library", "output.mp3")
