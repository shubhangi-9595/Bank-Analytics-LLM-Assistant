import sqlite3

from bank_assistant.data.bird import load_questions
from bank_assistant.rag.chunking import count_tokens
from bank_assistant.rag.evaluate import labelled_questions, score
from bank_assistant.rag.index import load_retriever, make_chunks
from bank_assistant.rag.labels import needed_terms, schema_from_db


def test_structured_chunks_one_per_dictionary_entry(bank):
    chunks = make_chunks(bank.knowledge_dir, "structured")
    keys = {c.metadata["key"] for c in chunks}
    assert {"account.frequency", "loan.status", "district.A11", "trans.k_symbol"} <= keys
    freq = next(c for c in chunks if c.metadata["key"] == "account.frequency")
    assert freq.text.startswith("[Accounts, clients and dispositions > account.frequency")  # contextual header


def test_fixed_chunks_respect_size(bank):
    assert all(count_tokens(c.text) <= 64 * 1.2 for c in make_chunks(bank.knowledge_dir, "fixed"))


def test_needed_terms_from_reference_sql(bank):
    schema = schema_from_db(sqlite3.connect(bank.db_path))
    sql = ("SELECT COUNT(T2.account_id) FROM district AS T1 JOIN account AS T2 ON T1.district_id = T2.district_id "
           "WHERE T1.A3 = 'east Bohemia' AND T2.frequency = 'POPLATEK PO OBRATU'")
    assert needed_terms(sql, schema) == {"A3 =", "'POPLATEK PO OBRATU' ="}
    # 'east Bohemia' is compared with A3, not a coded column, so it is not a dictionary term
    assert needed_terms("SELECT COUNT(*) FROM card WHERE type = 'gold'", schema) == {"'gold' ="}


def test_every_needed_term_exists_in_the_dictionary(bank):
    kb = "\n".join(c.text for c in make_chunks(bank.knowledge_dir))
    for item in labelled_questions(load_questions(bank.bird_questions), bank.db_path):
        assert all(t in kb for t in item.terms), item


def test_hybrid_retrieval_finds_code_meanings(bank):
    r = load_retriever(bank.index_dir, "hybrid", bank)
    top = [c.metadata["key"] for c, _ in r.retrieve("accounts with weekly statements", 2)]
    assert "account.frequency" in top
    items = labelled_questions(load_questions(bank.bird_questions), bank.db_path)
    assert score(r, items, k=4)["full_coverage"].value >= 2 / 3
