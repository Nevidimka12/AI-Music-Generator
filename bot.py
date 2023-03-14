import telebot
import setting  # API_KEY file

from telebot import types

bot = telebot.TeleBot(setting.API_KEY)


@bot.message_handler(content_types='text')
def text_reply(message):
    if message.text == 'Хочу отправить и удостовериться в получении':
        markup2 = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        item1 = types.KeyboardButton("Отправлю свой файл")
        item2 = types.KeyboardButton("Воспольуюсь твоим")
        markup2.add(item1, item2)
        bot.send_message(message.chat.id, 'Отлично! Можешь прислать мне любой mp3 файл. Если такого нет, я могу отправить тебе что-нибудь, что ты сможешь мне переслать:)', reply_markup=markup2)
        return
    elif message.text == 'Хочу получить mp3 файл':
        audio = open('some path', 'rb')
        bot.send_audio(message.chat.id, audio, 'воть')
        return
    elif message.text == 'Отправлю свой файл':
        bot.reply_to(message, 'жду :)')
        return
    elif message.text == 'Воспольуюсь твоим':
        audio = open('some path', 'rb')
        bot.send_audio(message.chat.id, audio, 'хм, попробуй это')
        return
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    item1 = types.KeyboardButton("Хочу отправить и удостовериться в получении")
    item2 = types.KeyboardButton("Хочу получить mp3 файл")
    markup.add(item1, item2)
    bot.send_message(message.chat.id, 'Привет! Я - тестовая версия бота. В данный момент я реализую две функции, выбери ниже ту, которую хочешь испытать)', reply_markup=markup)


@bot.message_handler(content_types='audio')
def audio_reply(message):
    bot.send_audio(message.chat.id, message.audio.file_id, 'возвращаю)')


bot.infinity_polling()
