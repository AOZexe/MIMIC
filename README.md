# MIMIC
IA que permite la opción de utilizar una voz personalizada, genera habla a partir de un texto dado y la voz elegida
Metodo de uso 
# Asigna un nombre para las muestras de audio y agrega minimo 3 muestras de audio de 1 minuto
CUSTOM_VOICE_NAME = "Claudia"

import os
from google.colab import files

custom_voice_folder = f"tortoise/voices/{CUSTOM_VOICE_NAME}"
os.makedirs(custom_voice_folder)
for i, file_data in enumerate(files.upload().values()):
  with open(os.path.join(custom_voice_folder, f'{i}.wav'), 'wb') as f:
    f.write(file_data)

 # Ingresa lo que quieres que diga
text = "Escucha la voz humana que imito "

# Selecciona el  tipos de voz "ultra_fast", "fast" (default), "standard", "high_quality"
preset = "high_quality"

# Ejecuta 
voice_samples, conditioning_latents = load_voice(CUSTOM_VOICE_NAME)
gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                          preset=preset)
torchaudio.save(f'generated-{CUSTOM_VOICE_NAME}.wav', gen.squeeze(0).cpu(), 24000)
IPython.display.Audio(f'generated-{CUSTOM_VOICE_NAME}.wav')
#Descargar audio:
from google.colab import files; files.download(f'generated-{CUSTOM_VOICE_NAME}.wav')


Es importante tener en cuenta que el programa solo acepta archivos con extensión ".wav" , para lograr una mayor fidelidad en la voz se recomienda el uso de audacity , y al momento de exportar el audio se recomienda una frecuencia de 22050 HZ 
