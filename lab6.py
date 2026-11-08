
import tkinter as tk
from tkinter import scrolledtext
import re
import random

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# ЛАБОРАТОРНА РОБОТА №6
# Тема: Розроблення інтелектуального чат-бота
# Предметна область: Туристичний асистент TravelBot
#
# Тип: гібридний ML-бот (класифікація намірів + правила)
# Технології: Python, scikit-learn, Tkinter, Regex
# ============================================================


# ============================================================
# 1. НАВЧАЛЬНИЙ ДАТАСЕТ
# 9 намірів, по 15 прикладів для кожного
# ============================================================

intents = {
    "greeting": [
        "Привіт",
        "Добрий день",
        "Вітаю",
        "Доброго ранку",
        "Добрий вечір",
        "Привіт бот",
        "Хай",
        "Здоров",
        "Радий тебе бачити",
        "Вітання",
        "Привіт туристичний асистент",
        "Доброго дня",
        "Здрастуй",
        "Привіт як справи",
        "Вітаю тебе бот"
    ],

    "destination": [
        "Куди поїхати у відпустку",
        "Порадь країну для подорожі",
        "Яке місто обрати для відпочинку",
        "Куди можна поїхати",
        "Порекомендуй напрямок",
        "Де краще відпочити",
        "Хочу поїхати у подорож",
        "Підкажи цікаве місто",
        "Яку країну відвідати",
        "Куди вирушити на вихідні",
        "Допоможи вибрати напрямок",
        "Порадь місце для відпочинку",
        "Я хочу відвідати Європу",
        "Які міста підходять для туризму",
        "Куди краще поїхати влітку"
    ],

    "transport": [
        "Як дістатися до Парижа",
        "Яким транспортом поїхати",
        "Як доїхати до Рима",
        "Чи можна поїхати автобусом",
        "Який транспорт обрати",
        "Як подорожувати потягом",
        "Порадь транспорт для поїздки",
        "Як найкраще добратися",
        "Чи є автобус до Праги",
        "Як доїхати до Варшави",
        "Що краще автобус чи поїзд",
        "Які є варіанти транспорту",
        "Чи можна полетіти літаком",
        "Який транспорт найдешевший",
        "Як дістатися до міста"
    ],

    "accommodation": [
        "Де зупинитися у Парижі",
        "Порадь готель",
        "Де знайти житло",
        "Як забронювати готель",
        "Де переночувати у Римі",
        "Яке житло обрати",
        "Підкажи хороший хостел",
        "Де можна жити під час подорожі",
        "Хочу знайти апартаменти",
        "Порадь місце проживання",
        "Як вибрати готель",
        "Які є варіанти житла",
        "Де знайти дешевий хостел",
        "Порадь бюджетне проживання",
        "Яке житло краще для туриста"
    ],

    "budget": [
        "Скільки коштує подорож",
        "Який бюджет потрібен",
        "Скільки грошей взяти",
        "Чи дорого подорожувати",
        "Як заощадити у подорожі",
        "Порадь бюджетну поїздку",
        "Яка вартість відпочинку",
        "Скільки витрачу на подорож",
        "Хочу дешеву подорож",
        "Як розрахувати бюджет",
        "Чи вистачить 500 євро",
        "Скільки коштує поїздка до Праги",
        "Мій бюджет 300 євро",
        "Хочу витратити 1000 доларів",
        "Скільки потрібно грошей на тиждень"
    ],

    "attractions": [
        "Що подивитися у Парижі",
        "Які є пам'ятки",
        "Куди сходити у Римі",
        "Що варто відвідати",
        "Порадь цікаві місця",
        "Які музеї подивитися",
        "Що цікавого у Празі",
        "Які туристичні місця є",
        "Що побачити у Варшаві",
        "Підкажи визначні пам'ятки",
        "Які місця відвідати туристу",
        "Куди піти на екскурсію",
        "Які визначні місця є у місті",
        "Що цікавого можна побачити",
        "Порадь туристичні пам'ятки"
    ],

    "documents": [
        "Які документи потрібні",
        "Чи потрібен закордонний паспорт",
        "Які документи взяти у подорож",
        "Чи потрібна віза",
        "Що потрібно для перетину кордону",
        "Які правила в'їзду",
        "Чи потрібне страхування",
        "Що перевірити перед виїздом",
        "Які вимоги до паспорта",
        "Які документи для Європи",
        "Чи можна їхати без візи",
        "Що потрібно для закордонної поїздки",
        "Які правила перетину кордону",
        "Які документи потрібні туристу",
        "Що взяти для поїздки за кордон"
    ],

    "duration": [
        "На скільки днів поїхати",
        "Скільки днів потрібно для подорожі",
        "Хочу поїхати на 5 днів",
        "Планую відпочинок на тиждень",
        "Скільки часу потрібно на Париж",
        "Чи вистачить трьох днів",
        "Порадь тривалість подорожі",
        "Хочу поїхати на вихідні",
        "Планую поїздку на 7 днів",
        "Скільки днів провести у Римі",
        "Маю 4 дні для відпочинку",
        "Скільки часу виділити на екскурсії",
        "Хочу подорожувати 10 днів",
        "Яка оптимальна тривалість поїздки",
        "Скільки днів варто залишитися"
    ],

    "goodbye": [
        "До побачення",
        "Бувай",
        "Дякую",
        "До зустрічі",
        "Гарного дня",
        "Дякую за допомогу",
        "Пока",
        "На все добре",
        "До наступного разу",
        "Бувай бот",
        "Дякую тобі",
        "Завершити розмову",
        "До скорої зустрічі",
        "Спасибі за консультацію",
        "На добраніч"
    ]
}


# ============================================================
# 2. СЦЕНАРІЇ ВИКОРИСТАННЯ (USE CASES)
# ============================================================

use_cases = {
    "UC1": "Привітання та початок діалогу",
    "UC2": "Вибір туристичного напрямку",
    "UC3": "Консультація щодо транспорту",
    "UC4": "Пошук варіантів проживання",
    "UC5": "Планування бюджету",
    "UC6": "Рекомендації туристичних пам'яток",
    "UC7": "Консультація щодо документів",
    "UC8": "Планування тривалості подорожі",
    "UC9": "Завершення діалогу"
}


# ============================================================
# 3. ПІДГОТОВКА ДАНИХ ТА НАВЧАННЯ ML-МОДЕЛІ
# ============================================================

texts = []
labels = []

for intent, examples in intents.items():
    for example in examples:
        texts.append(example.lower())
        labels.append(intent)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            lowercase=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1500,
            random_state=42
        )
    )
])

model.fit(texts, labels)

print("Модель успішно навчена!")
print("Кількість намірів:", len(intents))
print("Кількість навчальних прикладів:", len(texts))
print("Сценарії використання:", len(use_cases))


# ============================================================
# 4. БАЗА ЗНАНЬ
# Демонстраційні туристичні рекомендації
# ============================================================

knowledge = {
    "Париж": {
        "country": "Франція",
        "attractions": [
            "Ейфелева вежа",
            "Лувр",
            "Монмартр",
            "собор Нотр-Дам"
        ],
        "recommended_days": 4,
        "description": "Місто мистецтва, архітектури та музеїв."
    },

    "Рим": {
        "country": "Італія",
        "attractions": [
            "Колізей",
            "Пантеон",
            "Римський форум",
            "фонтан Треві"
        ],
        "recommended_days": 4,
        "description": "Історичне місто з античними пам'ятками."
    },

    "Прага": {
        "country": "Чехія",
        "attractions": [
            "Карлів міст",
            "Празький град",
            "Староміська площа",
            "Астрономічний годинник"
        ],
        "recommended_days": 3,
        "description": "Місто середньовічної архітектури."
    },

    "Варшава": {
        "country": "Польща",
        "attractions": [
            "Старе місто",
            "Королівський замок",
            "парк Лазєнки",
            "Палац культури і науки"
        ],
        "recommended_days": 3,
        "description": "Сучасна столиця з багатою історією."
    }
}


# ============================================================
# 5. СУТНОСТІ (ENTITIES)
# ============================================================

city_forms = {
    "Париж": ["париж", "парижа", "парижі", "парижу"],
    "Рим": ["рим", "рима", "римі", "риму"],
    "Прага": ["прага", "праги", "празі", "прагу"],
    "Варшава": [
        "варшава", "варшави", "варшаві", "варшаву"
    ]
}

country_forms = {
    "Франція": ["франція", "франції", "францію"],
    "Італія": ["італія", "італії", "італію"],
    "Чехія": ["чехія", "чехії", "чехію"],
    "Польща": ["польща", "польщі", "польщу"]
}

transport_forms = {
    "автобус": ["автобус", "автобусом", "автобуси"],
    "поїзд": ["поїзд", "поїздом", "потяг", "потягом"],
    "літак": ["літак", "літаком", "авіарейс"]
}


def find_dictionary_entity(text, dictionary):
    for canonical, forms in dictionary.items():
        for form in forms:
            pattern = r"(?<!\w)" + re.escape(form) + r"(?!\w)"
            if re.search(pattern, text):
                return canonical
    return None


def extract_entities(message):
    text = message.lower()
    entities = {}

    city = find_dictionary_entity(text, city_forms)
    country = find_dictionary_entity(text, country_forms)
    transport = find_dictionary_entity(text, transport_forms)

    if city:
        entities["city"] = city
        entities["country"] = knowledge[city]["country"]
    elif country:
        entities["country"] = country

    if transport:
        entities["transport"] = transport

    budget_match = re.search(
        r"(?<!\w)(\d+(?:[.,]\d+)?)\s*"
        r"(євро|eur|€|доларів|долари|usd|\$|грн|uah|гривень)",
        text
    )

    if budget_match:
        amount = float(budget_match.group(1).replace(",", "."))
        currency_text = budget_match.group(2)

        if currency_text in ["євро", "eur", "€"]:
            currency = "EUR"
        elif currency_text in [
            "доларів", "долари", "usd", "$"
        ]:
            currency = "USD"
        else:
            currency = "UAH"

        entities["budget"] = amount
        entities["currency"] = currency

    duration_match = re.search(
        r"(?<!\w)(\d+)\s*"
        r"(день|дні|днів|дня|добу|доби|діб|тиждень|тижні|тижнів)",
        text
    )

    if duration_match:
        number = int(duration_match.group(1))
        unit = duration_match.group(2)

        if unit.startswith("тиж"):
            number *= 7

        entities["days"] = number

    elif re.search(r"\bна тиждень\b", text):
        entities["days"] = 7

    elif re.search(r"\bна вихідні\b", text):
        entities["days"] = 2

    return entities


# ============================================================
# 6. КОНТЕКСТНА ПАМ'ЯТЬ
# ============================================================

def create_memory():
    return {
        "city": None,
        "country": None,
        "budget": None,
        "currency": None,
        "days": None,
        "transport": None,
        "last_intent": None,
        "history": []
    }


memory = create_memory()


def update_memory(entities):
    for key, value in entities.items():
        if key in memory:
            memory[key] = value


# ============================================================
# 7. КЛАСИФІКАЦІЯ НАМІРІВ
# ============================================================

def predict_intent(message):
    probabilities = model.predict_proba([message.lower()])[0]

    best_index = probabilities.argmax()
    confidence = float(probabilities[best_index])
    intent = model.classes_[best_index]

    if confidence < 0.18:
        return "unknown", confidence

    return intent, confidence


# ============================================================
# 8. КОНТЕКСТНІ УТОЧНЕННЯ
# ============================================================

def resolve_intent(message, intent, entities):
    lower = message.lower().strip()

    if lower in ["привіт", "вітаю", "добрий день"]:
        return "greeting"

    if lower in [
        "дякую", "спасибі", "бувай", "до побачення"
    ]:
        return "goodbye"

    if re.search(
        r"\b(що подивитися|що побачити|пам'ятки|музеї)\b",
        lower
    ):
        return "attractions"

    if re.search(
        r"\b(готель|хостел|житло|апартаменти)\b",
        lower
    ):
        return "accommodation"

    if re.search(
        r"\b(як дістатися|як доїхати|транспорт)\b",
        lower
    ):
        return "transport"

    if "budget" in entities and (
        intent in ["unknown", "destination", "duration"]
        or lower.startswith(("мій бюджет", "маю", "у мене"))
    ):
        return "budget"

    if "days" in entities and (
        intent in ["unknown", "destination"]
        or lower.startswith(("на ", "маю ", "планую на "))
    ):
        return "duration"

    if "city" in entities and (
        intent == "unknown"
        or lower in sum(city_forms.values(), [])
    ):
        previous = memory["last_intent"]

        if previous in [
            "transport",
            "accommodation",
            "attractions",
            "duration"
        ]:
            return previous

        return "destination"

    return intent


# ============================================================
# 9. ГЕНЕРАЦІЯ ВІДПОВІДЕЙ
# ============================================================

def generate_response(message):
    entities = extract_entities(message)

    predicted_intent, confidence = predict_intent(message)
    intent = resolve_intent(
        message, predicted_intent, entities
    )

    update_memory(entities)

    city = memory["city"]
    country = memory["country"]
    budget = memory["budget"]
    currency = memory["currency"]
    days = memory["days"]
    transport = memory["transport"]

    if intent == "greeting":
        response = (
            "Привіт! Я TravelBot — туристичний асистент. 🌍\n"
            "Допоможу обрати місто, транспорт, житло, "
            "бюджет, пам'ятки та тривалість подорожі.\n"
            "Куди хочеш поїхати?"
        )

    elif intent == "destination":
        if city and "city" in entities:
            data = knowledge[city]
            response = (
                f"{city}, {data['country']} — "
                f"цікавий напрямок для подорожі!\n"
                f"{data['description']}\n"
                f"Рекомендована тривалість: "
                f"{data['recommended_days']} дні(в).\n"
                "Можу розповісти про пам'ятки, житло "
                "або транспорт."
            )

        elif country and "country" in entities:
            cities = [
                name for name, data in knowledge.items()
                if data["country"] == country
            ]
            response = (
                f"Для подорожі до країни {country} "
                f"можу запропонувати: {', '.join(cities)}."
            )

        else:
            response = (
                "Рекомендую розглянути такі напрямки:\n"
                "🇫🇷 Париж — мистецтво та музеї.\n"
                "🇮🇹 Рим — історія та архітектура.\n"
                "🇨🇿 Прага — середньовічне місто.\n"
                "🇵🇱 Варшава — культура та історія.\n"
                "Напиши назву міста, яке тебе цікавить."
            )

    elif intent == "transport":
        if city:
            selected = (
                f"Ти обрав транспорт: {transport}.\n"
                if transport else ""
            )
            response = (
                f"{selected}"
                f"Для поїздки до міста {city} "
                "можна розглянути автобус, поїзд "
                "або літак залежно від місця відправлення.\n"
                "Для конкретного маршруту потрібне "
                "місто відправлення.\n"
                "Розклад і ціни перевіряй у перевізників."
            )
        else:
            response = (
                "До якого міста плануєш подорож?\n"
                "Наприклад: Як доїхати до Праги?"
            )

    elif intent == "accommodation":
        if city:
            response = (
                f"У місті {city} можна обрати:\n"
                "🏨 Готель — комфортне проживання.\n"
                "🛏️ Хостел — бюджетний варіант.\n"
                "🏠 Апартаменти — більше простору.\n"
                "Порівнюй розташування, відгуки, "
                "вартість та умови бронювання."
            )
        else:
            response = (
                "У якому місті шукаєш житло?\n"
                "Наприклад: Порадь житло у Римі."
            )

    elif intent == "budget":
        if budget is not None:
            # Орієнтовний розподіл, а не реальні ціни
            transport_part = budget * 0.30
            accommodation_part = budget * 0.35
            food_part = budget * 0.20
            entertainment_part = budget * 0.15

            response = (
                f"Твій бюджет: {budget:g} {currency}.\n"
                "Приклад розподілу бюджету:\n"
                f"🚆 Транспорт: {transport_part:.2f} {currency}\n"
                f"🏨 Житло: {accommodation_part:.2f} {currency}\n"
                f"🍽️ Харчування: {food_part:.2f} {currency}\n"
                f"🎭 Розваги: {entertainment_part:.2f} {currency}\n"
                "Це умовний розподіл, а не прогноз цін."
            )

            if days:
                response += (
                    f"\nСередній бюджет на день: "
                    f"{budget / days:.2f} {currency}."
                )
        else:
            response = (
                "Напиши бюджет своєї подорожі.\n"
                "Наприклад: Мій бюджет 500 євро."
            )

    elif intent == "attractions":
        if city and city in knowledge:
            places = knowledge[city]["attractions"]
            response = (
                f"У місті {city} рекомендую відвідати:\n"
                + "\n".join(
                    f"• {place}" for place in places
                )
            )
        else:
            response = (
                "Для якого міста шукаєш пам'ятки?\n"
                "Наприклад: Що подивитися у Парижі?"
            )

    elif intent == "documents":
        response = (
            "Перед закордонною подорожжю перевір:\n"
            "• Термін дії закордонного паспорта.\n"
            "• Візові вимоги для свого громадянства.\n"
            "• Правила перетину кордону.\n"
            "• Умови медичного страхування.\n"
            "• Додаткові вимоги країни призначення.\n"
            "Актуальні правила перевіряй на "
            "офіційних державних ресурсах."
        )

    elif intent == "duration":
        if days:
            response = (
                f"Ти плануєш подорож на {days} днів.\n"
            )

            if city:
                recommended = knowledge[city][
                    "recommended_days"
                ]
                response += (
                    f"Для міста {city} орієнтовно "
                    f"рекомендую {recommended} дні(в).\n"
                )

            response += (
                "Раджу залишити час для відпочинку "
                "та непередбачених змін."
            )

        elif city:
            recommended = knowledge[city][
                "recommended_days"
            ]
            response = (
                f"Для знайомства з містом {city} "
                f"орієнтовно достатньо "
                f"{recommended} дні(в).\n"
                "Напиши, на скільки днів плануєш поїздку."
            )

        else:
            response = (
                "Напиши місто та кількість днів.\n"
                "Наприклад: Планую Париж на 5 днів."
            )

    elif intent == "goodbye":
        response = (
            "Дякую за спілкування! ✈️\n"
            "Бажаю чудової подорожі!"
        )

    else:
        response = (
            "Вибач, я не зовсім зрозумів запит.\n"
            "Можу допомогти з напрямком, транспортом, "
            "житлом, бюджетом, пам'ятками, "
            "документами та тривалістю поїздки."
        )

    # Збереження історії поточного діалогу
    memory["history"].append({
        "user": message,
        "intent": intent,
        "confidence": round(confidence, 3),
        "entities": entities.copy(),
        "bot": response
    })

    if intent not in ["greeting", "goodbye", "unknown"]:
        memory["last_intent"] = intent

    return response


# ============================================================
# 10. ГРАФІЧНИЙ ІНТЕРФЕЙС TKINTER
# ============================================================

root = tk.Tk()
root.title("TravelBot — Інтелектуальний туристичний асистент")
root.geometry("780x720")
root.minsize(550, 500)
root.configure(bg="#1e293b")

title_label = tk.Label(
    root,
    text="🌍 TRAVEL BOT",
    font=("Arial", 25, "bold"),
    fg="white",
    bg="#1e293b"
)
title_label.pack(pady=(20, 5))

subtitle_label = tk.Label(
    root,
    text="Інтелектуальний асистент для планування подорожей",
    font=("Arial", 12),
    fg="#94a3b8",
    bg="#1e293b"
)
subtitle_label.pack(pady=(0, 15))

chat_area = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    font=("Arial", 12),
    bg="#334155",
    fg="white",
    insertbackground="white",
    padx=15,
    pady=15,
    state="disabled"
)
chat_area.pack(
    fill=tk.BOTH,
    expand=True,
    padx=20,
    pady=10
)

chat_area.tag_config(
    "user",
    foreground="#93c5fd",
    spacing3=12
)

chat_area.tag_config(
    "bot",
    foreground="#86efac",
    spacing3=16
)


# ============================================================
# 11. ОБРОБКА ПОВІДОМЛЕНЬ
# ============================================================

def add_message(sender, message, tag):
    chat_area.config(state="normal")
    chat_area.insert(
        tk.END,
        f"{sender}:\n{message}\n\n",
        tag
    )
    chat_area.config(state="disabled")
    chat_area.see(tk.END)


def send_message(event=None):
    message = message_entry.get().strip()

    if not message:
        return

    message_entry.delete(0, tk.END)

    add_message("Ви", message, "user")

    try:
        response = generate_response(message)
    except Exception as error:
        response = (
            "Виникла помилка під час обробки повідомлення."
        )
        print("Помилка:", error)

    add_message("TravelBot", response, "bot")


def clear_chat():
    memory.clear()
    memory.update(create_memory())

    chat_area.config(state="normal")
    chat_area.delete("1.0", tk.END)
    chat_area.config(state="disabled")

    add_message(
        "TravelBot",
        "Привіт! Я TravelBot. 🌍\n"
        "Допоможу спланувати твою подорож.\n"
        "Наприклад, запитай: Що подивитися у Парижі?",
        "bot"
    )


# ============================================================
# 12. ПОЛЕ ВВЕДЕННЯ ТА КНОПКИ
# ============================================================

input_frame = tk.Frame(
    root,
    bg="#1e293b"
)
input_frame.pack(
    fill=tk.X,
    padx=20,
    pady=10
)

message_entry = tk.Entry(
    input_frame,
    font=("Arial", 13),
    bg="#475569",
    fg="white",
    insertbackground="white",
    relief="flat"
)
message_entry.pack(
    side=tk.LEFT,
    fill=tk.X,
    expand=True,
    ipady=12,
    padx=(0, 10)
)

send_button = tk.Button(
    input_frame,
    text="Надіслати",
    command=send_message,
    font=("Arial", 11, "bold"),
    fg="white",
    bg="#2563eb",
    relief="flat",
    padx=15,
    pady=10,
    cursor="hand2"
)
send_button.pack(side=tk.RIGHT)

clear_button = tk.Button(
    root,
    text="Очистити чат",
    command=clear_chat,
    font=("Arial", 10),
    fg="white",
    bg="#64748b",
    relief="flat",
    padx=15,
    pady=8,
    cursor="hand2"
)
clear_button.pack(pady=(0, 15))

message_entry.bind("<Return>", send_message)

clear_chat()
message_entry.focus_set()

root.mainloop()


