import os

import churro.providers.openai_provider as openai_mod
from churro.providers.openai_provider import DEFAULT_MODEL, OpenAIProvider
from churro.providers.provider import (
    APIRequestError,
    AuthenticationError,
    MissingAPIKeyError,
    UnexpectedResponseError,
)
from churro.core.session import SessionState, create_session


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content="Fake reply"):
        self.choices = [FakeChoice(content)]


class FakeEmptyChoices:
    def __init__(self):
        self.choices = []


class FakeNoChoices:
    pass


class FakeCompletions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.calls.append((kwargs.get("model"), kwargs.get("messages")))
        item = self.owner.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeChat:
    def __init__(self, owner):
        self.completions = FakeCompletions(owner)


class FakeClient:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses if responses is not None else [FakeCompletion()]

    @property
    def chat(self):
        return FakeChat(self)


class _FakeNetworkError(Exception):
    pass


class _FakeAuthError(Exception):
    pass


def make_provider(responses=None, **kwargs):
    return OpenAIProvider(client=FakeClient(responses=responses), **kwargs)


def test_sends_expected_messages():
    provider = make_provider(model="gpt-4o")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    provider.send(messages)

    model, passed = provider._client.calls[0]
    assert passed == messages


def test_returns_model_text():
    provider = make_provider(responses=[FakeCompletion("I found the bug.")])
    reply = provider.send([{"role": "user", "content": "hi"}])
    assert reply == "I found the bug."


def test_correct_model_passed_to_api():
    provider = make_provider(model="gpt-4o-mini")
    provider.send([{"role": "user", "content": "hi"}])

    model, _ = provider._client.calls[0]
    assert model == "gpt-4o-mini"


def test_default_model_is_documented():
    assert DEFAULT_MODEL == "gpt-4o"


def test_missing_api_key_handled():
    saved = os.environ.get("OPENAI_API_KEY")
    os.environ.pop("OPENAI_API_KEY", None)
    try:
        try:
            OpenAIProvider(model="gpt-4o")
        except MissingAPIKeyError:
            pass
        else:
            raise AssertionError("expected MissingAPIKeyError")
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved


def test_api_failure_handled_cleanly():
    original = openai_mod.APIConnectionError
    openai_mod.APIConnectionError = _FakeNetworkError
    try:
        provider = make_provider(responses=[_FakeNetworkError("boom")])
        try:
            provider.send([{"role": "user", "content": "hi"}])
        except APIRequestError:
            pass
        else:
            raise AssertionError("expected APIRequestError for network failure")
    finally:
        openai_mod.APIConnectionError = original


def test_authentication_failure_handled_cleanly():
    original = openai_mod._OpenAIAuthenticationError
    openai_mod._OpenAIAuthenticationError = _FakeAuthError
    try:
        provider = make_provider(responses=[_FakeAuthError("bad key")])
        try:
            provider.send([{"role": "user", "content": "hi"}])
        except AuthenticationError:
            pass
        else:
            raise AssertionError("expected AuthenticationError")
    finally:
        openai_mod._OpenAIAuthenticationError = original


def test_unexpected_response_shape_handled():
    provider = make_provider(responses=[FakeNoChoices()])
    try:
        provider.send([{"role": "user", "content": "hi"}])
    except UnexpectedResponseError:
        pass
    else:
        raise AssertionError("expected UnexpectedResponseError for empty choices")


def test_empty_choices_response_handled():
    provider = make_provider(responses=[FakeEmptyChoices()])
    try:
        provider.send([{"role": "user", "content": "hi"}])
    except UnexpectedResponseError:
        pass
    else:
        raise AssertionError("expected UnexpectedResponseError for no choices")


def test_provider_does_not_modify_session_state():
    session = create_session(goal="Fix a bug", provider="openai", model="gpt-4o")
    session.state.status = "in_progress"
    before = session.model_dump()

    provider = make_provider(responses=[FakeCompletion("A reply that changes nothing.")])
    provider.send([{"role": "user", "content": "work on it"}])

    assert session.model_dump() == before
    assert session.state == SessionState(
        goal="Fix a bug",
        status="in_progress",
    )
    assert session.conversation_history == []


def test_provider_returns_plain_text_without_state_tags():
    provider = make_provider(
        responses=[FakeCompletion("Reply.\n<state_update>{}</state_update>")]
    )
    reply = provider.send([{"role": "user", "content": "hi"}])
    assert reply == "Reply.\n<state_update>{}</state_update>"


TEST_FUNCTIONS = [
    test_sends_expected_messages,
    test_returns_model_text,
    test_correct_model_passed_to_api,
    test_default_model_is_documented,
    test_missing_api_key_handled,
    test_api_failure_handled_cleanly,
    test_authentication_failure_handled_cleanly,
    test_unexpected_response_shape_handled,
    test_empty_choices_response_handled,
    test_provider_does_not_modify_session_state,
    test_provider_returns_plain_text_without_state_tags,
]


def main() -> None:
    failures = 0
    for fn in TEST_FUNCTIONS:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        else:
            print(f"PASS  {fn.__name__}")
    if failures:
        raise SystemExit(f"\n{failures} test(s) failed")
    print(f"\nAll {len(TEST_FUNCTIONS)} tests passed.")


if __name__ == "__main__":
    main()