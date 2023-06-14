# AI-Music-Generator

Проект AI-Music-Generator реализует распознавание мелодии и аккордов в заданной песне, а также предсказывает новые аккорды и мелодию на основе предыдущих данных. Весь функционал обёрнут в интерфейс Telegram бота. Программа работает на версии Python >= 3.10, главный файл - bot.py.

Описание задач каждого файла:
  - bot.py: Собственно, сам бот.
  - save_transposed_audio: Определение тональности исходной песни (krumhansl-schmuckler key-finding algorithm) и последующая конвертация в до мажор/ля минор для улучшенной работы моделей.
  - chord_classifier.py: Извлечение аккордов из песни с помощью vamp плагина Chordino и последующим переводом этих данных в формат {название_аккорда, длительность}.
  - melody_classifier.py: Извлечение мелодии с помощью функции PredominantPitchMelodia библиотеки Essentia, последующей обработкой и переводом в формат {нота, длительность}.
  - generation.py: Предсказание новых аккордов и мелодии и их длительностей с помощью моделей Catboost и LSTM (а также многочисленными преобразованиями к нужному формату). Модели берутся из соответствующих файлов, кодировка и декодировка происходит через Code/Decode.json. 
  - midi_saver.py: Объединение форматов {название_аккорда, длительность} и {нота, длительность} и конвертация в итоговый midi файл (Дорожка аккордов - 0, мелодии - 1). 
  
