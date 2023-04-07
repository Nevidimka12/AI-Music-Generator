import telebot
import setting  # API_KEY file
import requests
import chord_classifier
import midi_saver
import os
import generation
from telebot import types
import random

bot = telebot.TeleBot(setting.API_KEY)


@bot.message_handler(content_types=['text'])
def text_reply(message):
    if message.text == 'Хочу отправить свой аудиофайл и получить его продолжение':
        bot.send_message(message.chat.id, 'Отлично! Можешь прислать мне любой аудиофайл для обработки')
        return
    elif message.text == 'Хочу получить аудиофайл для тестирования':
        files = os.listdir('/Users/timurabdulkadirov/Desktop/папка-отправка')
        n = random.randint(0, len(files)-1)
        audio = open('/Users/timurabdulkadirov/Desktop/папка-отправка/'+files[n], 'rb')
        bot.send_audio(message.chat.id, audio)
        audio.close()
        return
    elif message.text == 'Хочу получить wav файл':
        audio = open('result_rec.wav', 'rb')
        bot.send_audio(message.chat.id, audio)
        audio.close()
        return
    elif message.text == 'Хочу получить mid файл':
        audio = open('result.mid', 'rb')
        bot.send_audio(message.chat.id, audio)
        audio.close()
        return
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    item1 = types.KeyboardButton("Хочу отправить свой аудиофайл и получить его продолжение")
    item2 = types.KeyboardButton("Хочу получить аудиофайл для тестирования")
    markup.add(item1, item2)
    bot.send_message(message.chat.id, 'Привет! Я - тестовая версия бота. В данный момент я реализую две функции, выбери ниже ту, которую хочешь испытать)', reply_markup=markup)


@bot.message_handler(content_types=['audio'])
def audio_reply(message):
    file_id = message.audio.file_id
    file_info = bot.get_file(file_id)
    file = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(setting.API_KEY, file_info.file_path))
    with open('aaa1.mp3', 'wb') as f:
        f.write(file.content)
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    item1 = types.KeyboardButton("Хочу получить wav файл")
    item2 = types.KeyboardButton("Хочу получить mid файл")
    markup.add(item1, item2)
    midi_saver.save_chords_as_midi(generation.predict_sequence(chord_classifier.transform_song_to_data('aaa1.mp3')[0], n=5, predicted_chord_durations=1.337))
    os.system('timidity result.mid -Ow -o result_rec.wav')
    bot.send_message(message.chat.id, 'Выбери, какой файл получить:', reply_markup=markup)


bot.infinity_polling()
