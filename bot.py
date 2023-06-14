import telebot
import melody_classifier
import setting  # API_KEY file
import requests
import chord_classifier
import midi_saver
import os
import generation
from telebot import types
import random
import save_transposed_audio
import subprocess

bot = telebot.TeleBot(setting.API_KEY)

c = False
key_glob = 0
digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
duration = 0


def text_sub(chat_id):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    item1 = types.KeyboardButton("Хочу отправить свой аудиофайл")
    item2 = types.KeyboardButton("Хочу получить аудиофайл")
    markup.add(item1, item2)
    bot.send_message(chat_id, 'Привет! Выбери ниже функцию, которую хочешь испытать)', reply_markup=markup)


def reply_to(chat_id, type):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    item1 = types.KeyboardButton("Получить wav файл")
    item2 = types.KeyboardButton("Получить mid файл")
    item3 = types.KeyboardButton("Получить mp3 файл")
    item4 = types.KeyboardButton("Вернуться к началу")
    markup.add(item1, item2, item3, item4)

    name_song = 'song_transposed.mp3' if not c else 'voice_transposed.mp3'
    melody_data = melody_classifier.get_melody_data(name_song)
    chord_data = chord_classifier.transform_song_to_data(name_song)

    if type == 'chords':
        chord_data = generation.predict_sequence(chord_data, duration)
    else:
        melody_data += generation.save_predict_melody(name_song, duration)

    midi_saver.save_song_as_midi(chord_data, melody_data, BPM=chord_classifier.detect_BPM(name_song),
                                 transpose=-key_glob, melody_volume=100, chords_volume=30)

    os.system('timidity result.mid -Ow -o result_rec.wav')
    os.system('timidity result.mid -Ow -o result_rec.mp3')
    bot.send_message(chat_id, 'Выбери, какой файл получить:', reply_markup=markup)


@bot.message_handler(content_types=['text'])
def text_reply(message):
    if message.text[0] in digits:
        global duration
        duration = int(message.text)
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        item1 = types.KeyboardButton("Дополнить аккорды")
        item2 = types.KeyboardButton("Дополнить мелодию")
        markup.add(item1, item2)
        bot.send_message(message.chat.id, 'Выбери продолжение', reply_markup=markup)
        return
    elif message.text == 'Дополнить аккорды':
        reply_to(message.chat.id, 'chords')
        return
    elif message.text == 'Дополнить мелодию':
        reply_to(message.chat.id, 'melody')
        return
    elif message.text == 'Хочу отправить свой аудиофайл':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        item1 = types.KeyboardButton("Вернуться к началу")
        markup.add(item1)
        bot.send_message(message.chat.id, 'Отлично! Можешь прислать мне любой аудиофайл для обработки', reply_markup=markup)
        return
    elif message.text == 'Хочу получить аудиофайл':
        files = os.listdir('/Users/timurabdulkadirov/Desktop/папка-отправка')
        n = random.randint(0, len(files) - 1)
        audio = open('/Users/timurabdulkadirov/Desktop/папка-отправка/' + files[n], 'rb')
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        item1 = types.KeyboardButton("Вернуться к началу")
        markup.add(item1)
        bot.send_audio(message.chat.id, audio, reply_markup=markup)
        audio.close()
        return
    elif message.text == 'Получить mp3 файл':
        audio = open('result_rec.mp3', 'rb')
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        item1 = types.KeyboardButton("Вернуться к началу")
        markup.add(item1)
        bot.send_audio(message.chat.id, audio, reply_markup=markup)
        audio.close()
        return
    elif message.text == 'Получить wav файл':
        audio = open('result_rec.wav', 'rb')
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        item1 = types.KeyboardButton("Вернуться к началу")
        markup.add(item1)
        bot.send_audio(message.chat.id, audio, reply_markup=markup)
        audio.close()
        return
    elif message.text == 'Получить mid файл':
        audio = open('result.mid', 'rb')
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        item1 = types.KeyboardButton("Вернуться к началу")
        markup.add(item1)
        bot.send_audio(message.chat.id, audio, reply_markup=markup)
        audio.close()
        return
    elif message.text == 'Вернуться к началу':
        text_sub(message.chat.id)
        return
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    item1 = types.KeyboardButton("Хочу отправить свой аудиофайл")
    item2 = types.KeyboardButton("Хочу получить аудиофайл")
    markup.add(item1, item2)
    bot.send_message(message.chat.id, 'Привет! Выбери ниже функцию, которую хочешь испытать)', reply_markup=markup)


@bot.message_handler(content_types=['audio'])
def audio_reply(message):
    global c
    c = False
    file_id = message.audio.file_id
    file_info = bot.get_file(file_id)
    file = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(setting.API_KEY, file_info.file_path))
    with open('song.mp3', 'wb') as f:
        f.write(file.content)
        exit_code, key, accuracy = save_transposed_audio.transpose_song('song.mp3', 'song_transposed.mp3', 0.5)
    global key_glob
    key_glob = key

    bot.send_message(message.chat.id, 'Выбери длительность продления песни в секундах')


@bot.message_handler(content_types=['voice'])
def get_audio_messages(message: types.Message):
    global c
    c = True
    file_id = message.voice.file_id
    file_info = bot.get_file(file_id)
    file = bot.download_file(file_info.file_path)
    file_name = 'voice.ogg'
    with open(file_name, 'wb') as f:
        f.write(file)
    os.remove('voice.mp3')
    subprocess.run(["ffmpeg", "-i", "voice.ogg", "voice.mp3"])
    exit_code, key, accuracy = save_transposed_audio.transpose_song('voice.mp3', 'voice_transposed.mp3', 0.5)
    global key_glob
    key_glob = key

    bot.send_message(message.chat.id, 'Выбери длительность продления песни в секундах')


bot.infinity_polling()
