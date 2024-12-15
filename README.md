# llm_history_agent
Мини Улиточка на страже истории! Выпускной проект курса "Введение в LLM".

## Общая идея проекта
Наш агент предполагался как помощник ученику в освоении такого непростого предмета как история. В силу дискуссионности
исторических подходов в современном моменте критически важно опираться на надёжные источники. По счастливой случайности
один из участников проекта (Макаров Игорь) обладает обширной экспертизой в предмете "История России", потому было
принято решение выбрать не целиком науку, а одну из самых не тривиальных тем для учебного проекта - Русско-Японскую
войну.

Один из самых серьёзных источников по теме - лекции Фёдора Викторовича Лисицина на youtube, поэтому было принято решение
отталкиваться от оцифровки лекций - лекции ученикам смотреть долго и скучно, а бот быстро найдёт нужную информацию.

1. RAG, основанный на транскрибации видеозаписей по теме, т.к. оригинальный, точный и очень обширный источник.
2. TG интерфейс, как наиболее простой способ коммуникации с учениками.
3. API GigaChat & GigaChatEmbeddings, потому что закуплено много токенов :)

## Актуальный стек и архитектура
![LLM_hist.png](artifacts%2FLLM_hist.png)

1. Whisper - транскрибация видео.
2. GigaChatEmbeddings - эмбеддер.
3. Aiogram - TG бот.
4. Langchain - обвязка RAG в составе:
   1. Document loader
   2. RecursiveCharacterTextSplitter
   3. Интеграция c ChromaDB
4. ChromaDB - векторное хранилище
5. Docker - деплой.
6. GigaChatPro - LLM.

## Структура проекта
1. app - сам сервис
   1. routes - руты на бота, команды и генерация.
   2. scripts - загрузчик документов, наполнение инстанса ChromaDB.
   3. services - основные сервисы бота:
      1. chroma_service - сервис работы с ChromaDB с полным набором функций.
      2. embeddings - сервис эмбеддера
      3. gigachat_service - сервис генерации. Здесь же хранится system_prompt
      4. rag_service - ядро рага, функции загрузки документов и генерации ответа
      5. text_processor - набор функций для обработки документа - чтение, очиста, чанкирование и т.д.
   4. tests - базовый набор тестов сервиса.
   5. utils - стартовые команды бота и машина состояний.
   6. app.py - стартовый файл бота.
   7. config.py - красивый файл настроек.
2. scratches - утилитарная пака препроцессинга
   1. cleaned_docs - директория с очищенными файлами для RAGa. Качетсво файлов 6-7 из 10, но это очень трудоёмкая работа по выверке (на имеющиеся файлы потрачено более 14 часов только на ручную проверку).
   2. transcribitions.py - скрипт whisper.
3. .env.example - пример файла настроек для локального запуска бота.
4. Остальные файлы не нуждаются в пояснениях.

## Пример работы
![example.png](artifacts%2Fexample.png)

## Известные проблемы
1. Качество выдаваемых сслыок на источники - несмотря на то, что источники выдаются большей частью хорошие, ссылки на них чаще всего - невалидное. С проблемой боремся.
2. To_do поиск имеющегося документа в ChromaDB (chroma_service)
3. To_do вынос функции загрузки документов в отдельный сервис (rag_service)
4. Отсутствует автоматическая валидация RAG, ответы валидируются экспертом вручную.
5. Отсутствует выдача метаданных с ответом LLM.

## Заключение
По экспертной оценке (Игорь Макаров) бот выдаёт очень хорошие ответы на вопросы по теме. Вопросы не по теме ботом успешно игнорируются.
Бота можно использовать как прототип современной системы обучения предмету "История России" при наполнении контентом.

### Авторы

![mini.jpg](artifacts%2Fmini.jpg)

[Игорь Макаров](https://t.me/qeshtir)

[Екатерина Костюк](https://t.me/Jenova_13)

[Павел Соколов](https://t.me/SPI_q)

### Лицензия

Этот проект распространяется по лицензии MIT.

### Благодарность

[AI Talent Hub](https://ai.itmo.ru/)