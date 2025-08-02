from core.media_processor import MediaProcessor
from core.tts_processor import TTSProcessor

if __name__ == '__main__':
    media_processor = MediaProcessor()
    media_processor.generate_reference_audio(r'C:\Users\leica\Desktop\1\output\1_vocal.wav',
                                             r'C:\Users\leica\Desktop\1\output\1_processed.srt')
    tts_processor = TTSProcessor()
    tts_processor.generate_tts_from_reference(
        r'C:\Users\leica\Desktop\1\output\reference_audio\1_vocal_reference_audio_results.json')
