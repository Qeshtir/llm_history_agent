import json
import logging
from langchain import hub
from langchain_openai import ChatOpenAI
from langsmith import Client
from services.rag_service import RAGService  # Импорт вашего класса RAGService
from config import settings

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_evaluation.log"),  # Лог сохраняется в файл
        logging.StreamHandler(),  # Дублирование в консоль
    ],
)
logger = logging.getLogger(__name__)

# Инициализация RAG-сервиса
rag_service = RAGService()
collection_name = (
    settings.COLLECTION_NAME
)  # Замените на актуальное название вашей коллекции

# Инициализация LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY)
langsmith_client = Client(api_key=settings.LANGSMITH_API_KEY)

# Загрузка шаблона для оценки
grade_prompt_answer_accuracy = hub.pull("langchain-ai/rag-answer-vs-reference")
answer_grader = grade_prompt_answer_accuracy | llm


def evaluate_rag_answers(test_data_path: str, collection_name: str) -> None:
    """
    Тестирует точность ответов RAG-сервиса, сравнивая их с эталонными ответами.

    Args:
        test_data_path (str): Путь к файлу с тестовыми данными.
        collection_name (str): Название коллекции документов.
    """
    # Загрузка тестовых данных
    try:
        with open(test_data_path, "r", encoding="utf-8") as file:
            test_data = json.load(file)
    except FileNotFoundError:
        logger.error(f"Файл {test_data_path} не найден.")
        return

    # Начало трека эксперимента
    experiment_name = "RAG Answer Evaluation"
    langsmith_client.start_experiment(experiment_name)

    for example in test_data:
        question = example["question"]
        expected_answer = example["expected_answer"]

        try:
            # Генерация ответа с помощью RAG
            generated_answer = rag_service.generate_answer(question, collection_name)

            # Оценка ответа
            score = answer_grader.invoke(
                {
                    "question": question,
                    "correct_answer": expected_answer,
                    "student_answer": generated_answer,
                }
            )
            evaluation_score = score["Score"]

            # Логирование результатов
            logger.info(f"Вопрос: {question}")
            logger.info(f"Эталонный ответ: {expected_answer}")
            logger.info(f"Ответ сервиса: {generated_answer}")
            logger.info(f"Оценка: {evaluation_score}")
            logger.info("-" * 50)

            # Отправка данных в LangSmith
            langsmith_client.log_run(
                experiment_name=experiment_name,
                inputs={"question": question, "expected_answer": expected_answer},
                outputs={"generated_answer": generated_answer},
                metrics={"accuracy_score": evaluation_score},
            )

        except Exception as e:
            logger.error(f"Ошибка при обработке вопроса '{question}': {str(e)}")
            langsmith_client.log_run(
                experiment_name=experiment_name,
                inputs={"question": question},
                outputs={"error": str(e)},
                metrics={"accuracy_score": 0},
            )

    # Завершение эксперимента
    langsmith_client.end_experiment(experiment_name)


if __name__ == "__main__":
    # Укажите путь к файлу с тестовыми данными
    test_data_path = "data/test_data.json"  # Файл с тестовыми данными

    # Запустите оценку
    evaluate_rag_answers(test_data_path, collection_name)
