DE_KEY = "de"
EN_KEY = "en"

REPHRASE_PROMPT_EN = """
    The latest user input was:
    {user_input}

    Generate a reply to return to the user based on these actions you are supposed to perform next:
    {responses}

    Use the text parameter provided with the utter actions as a basis for what you're supposed to say. 
    - If "rephrase" is True, you're allowed to rephrase the text to make it sound more natural. If a rephrase prompt is available, consider them like instructions.
    - If rephrase is False, output the text exactly as defined for the utter action.
    - If there are multiple utter actions, ensure that you are generating a cohesive reply including all of them. Avoid rephrasing for utter actions with "rephrase" False.
    Only add a greeting if this is the first bot message in the conversation history.

    Now generate your reply.
"""

REPHRASE_PROMPT_DE = """
    Die letzte Nachricht des Nutzers war:
    {user_input}

    Generiere eine Antwort an den nutzer basierend auf den Aktionen, die als nächstes ausgeführt werden sollen:
    {responses}

    Nutze den "text" parameter in den utter_ Aktionen als Basis dafür, was du sagen sollst.
    - Falls "rephrase" auf True gesetzt ist, darfst du die Antwort frei formulieren, sodass sie so natürlich wie möglich klingt. Wenn es einen rephrase prompt gibt, nehme diese als Anleitung zur Hilfe.
    - Falls "rephrase" auf False gesetzt ist, muss die Antwort genauso wiedergegeben werden wie sie im "text" parameter steht.
    - Falls es mehrere utter Aktionen gibt, generiere eine zusammenhängende Antwort aus den gegebenen utter Aktionen. Beachte aber nichts bei utter Aktionen zu ändern, bei denen "rephrase" False ist.
    Begrüße den Nutzer nur dann, wenn es in der Historie noch keine Nachricht an den Nutzer gibt.

    Generiere deine Antwort auf Deutsch.
"""

REPHRASE_PROMPTS = {
    EN_KEY: REPHRASE_PROMPT_EN,
    DE_KEY: REPHRASE_PROMPT_DE
}