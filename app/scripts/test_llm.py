import json
import logging
from langchain import hub
from langchain_gigachat import GigaChat
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langsmith.evaluation import evaluate
from langsmith import Client
from services.rag_service import RAGService
from config import settings
import os

# Настройка логгера
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Задает переменные окружения, согласно документации LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "test_langchain"

# Инициализация RAG-сервиса
rag_service = RAGService()
collection_name = settings.COLLECTION_NAME

# Загрузка шаблона для оценки
grade_prompt_answer_accuracy = hub.pull("langchain-ai/rag-answer-vs-reference")

# Загрузка llm-оценщика
llm = ChatMistralAI(model="mistral-large-2411", api_key=settings.MISTRAL_API_KEY)

# Загрузка клиента LangSmith
langsmith_client = Client(api_key=settings.LANGSMITH_API_KEY)

# Цепочка оценки
answer_grader = grade_prompt_answer_accuracy | llm


dataset_name = "History RAG Queries Example"


def create_dataset(data_path: str, client: Client) -> None:
    """
    Создает или использует существующий датасет на основе предоставленных тестовых данных.

    Параметры:
    - test_data_path (str): Путь к файлу с тестовыми данными в формате JSON.
    - client (object): Клиент для взаимодействия с системой управления датасетами.

    Возвращаемое значение:
    Функция не возвращает значения. В случае ошибки завершает выполнение.
    """
    try:
        with open(data_path, "r", encoding="utf-8") as file:
            test_data = json.load(file)
    except FileNotFoundError:
        logger.error(f"Файл {test_data_path} не найден.")
        return
    except json.JSONDecodeError:
        logger.error(f"Ошибка при чтении JSON файла {test_data_path}.")
        return

    # Проверка существования датасета
    datasets = list(client.list_datasets())
    dataset_exists = any(dataset.name == dataset_name for dataset in datasets)

    if dataset_exists:
        logger.info(f"Dataset '{dataset_name}' already exists. Using existing dataset.")
        dataset = next(dataset for dataset in datasets if dataset.name == dataset_name)
    else:
        logger.info(f"Dataset '{dataset_name}' does not exist. Creating new dataset.")
        dataset = client.create_dataset(dataset_name=dataset_name)

    inputs, outputs = zip(
        *[
            ({"question": item["question"]}, {"ground_truth": item["ground_truth"]})
            for item in test_data
        ]
    )
    client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)
    logger.info("Test DB created successfully!")


def answer_evaluator(run: object, example: dict) -> dict:
    """
    Простой оценщик точности ответов RAG.

    Параметры:
    - run (object): Объект, содержащий информацию о выполнении RAG-сервиса.
    - example (dict): Пример, содержащий вопрос и эталонный ответ.

    Возвращаемое значение:
    - dict: Словарь с ключом 'answer_v_reference_score', содержащий оценку и объяснение.
    """

    # Получаем вопрос, валидационный ответ и ответ системы
    input_question = example.inputs["question"]
    reference = example.outputs["ground_truth"]
    prediction = run.outputs["answer"]

    # оцениваем скор
    score_result = answer_grader.invoke(
        {
            "question": input_question,
            "correct_answer": reference,
            "student_answer": prediction,
        }
    )
    score = score_result["Score"]
    explanation = score_result["Explanation"]

    return {
        "key": "answer_v_reference_score",
        "score": score,
        "explanation": explanation,
    }


def predict_rag_answer(example: dict) -> dict:
    """
    Генерирует ответ RAG-сервиса на основе заданного вопроса.

    Параметры:
    - example (dict): Пример, содержащий вопрос.

    Возвращаемое значение:
    - dict: Словарь с ответом RAG-сервиса.
    """
    response = rag_service.generate_answer(example["question"], collection_name)
    return {"answer": response}


def evaluate_rag_answers() -> None:
    """
    Тестирует точность ответов RAG-сервиса, сравнивая их с эталонными ответами.

    Параметры:
    Функция не принимает параметров.

    Возвращаемое значение:
    Функция не возвращает значения.
    """

    experiment_results = evaluate(
        predict_rag_answer,
        data=dataset_name,
        evaluators=[answer_evaluator],
        experiment_prefix="rag-answer-v-reference",
    )


if __name__ == "__main__":

    # Файл с тестовыми данными
    test_data_path = "/app/data/test_data.json"

    # Создает тестовый датасет в LangSmith, если его нет
    create_dataset(test_data_path, langsmith_client)

    # Запускает оценку
    evaluate_rag_answers()
