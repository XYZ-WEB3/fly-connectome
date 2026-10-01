"""Смысл ролика: речь -> текст (Whisper, локально) -> словарь -> моменты «хорошо» / «плохо».

    import meaning as M
    ev = M.meaning_events(video)      # [{t, kind: 'pos'|'neg', word, text}] — кэш в tiktok/_cache/<имя>.words.json

По этим моментам экспериментатор (мы) даёт мухе сахар («хорошо») или горькое («плохо»).
Честность: муха слов не понимает; КОГДА дать каплю решаем мы по словарю, реакция — из её нейронов.
Словарь — tiktok/lexicon.json, его можно править руками (основы слов, без окончаний).
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEXICON = HERE / "lexicon.json"
CACHE = HERE / "_cache"
MODEL = "large-v3"                     # уже лежит в кэше huggingface (Systran/faster-whisper-large-v3)

DEFAULT_LEXICON = {
    "_как_править": "основы слов в нижнем регистре; слово ролика считается, если НАЧИНАЕТСЯ с основы",
    "pos": ["класс", "круто", "крут", "красав", "отличн", "супер", "молод", "любл", "любов", "люби", "счаст",
            "рад", "ура", "победа", "побед", "выигр", "лучш", "прекрасн", "шикарн", "кайф", "обожа", "спасибо",
            "смешн", "ахах", "хаха", "топ", "имба", "легенд", "вкусн", "сладк", "нрав", "хорош", "добр", "мил",
            "ништяк", "огонь", "респект", "поздравл", "good", "great", "love", "nice", "cool", "win", "best",
            "awesome", "happy", "yes"],
    "neg": ["плох", "ужас", "кошмар", "отстой", "ненави", "умер", "умир", "смерт", "убил", "убив", "убей", "боль",
            "больн", "груст", "печал", "плач", "плак", "слез", "страш", "страх", "жесть", "проигр", "провал",
            "дерьм", "говн", "хрен", "блин", "черт", "чёрт", "фу", "гадост", "мерз", "противн", "тупо", "туп",
            "дурак", "идиот", "лох", "позор", "обид", "зло", "злой", "бесит", "ненорм", "горьк", "кринж", "лаг",
            "bad", "hate", "die", "dead", "kill", "sad", "cry", "fail", "lose", "lost", "worst", "no", "damn"],
    "mat_neg": True,
}
MAT = re.compile(r"^(бля|блят|сук|пизд|хуй|хуе|хуя|еба|ёба|ебан|ебл|нахуй|пидор|мудак|fuck|shit)", re.I)


def load_lexicon():
    if not LEXICON.exists():
        LEXICON.write_text(json.dumps(DEFAULT_LEXICON, ensure_ascii=False, indent=1), encoding="utf8")
    return json.loads(LEXICON.read_text(encoding="utf8"))


def transcribe(video, verbose=True):
    """-> [{t0, t1, word}] по словам. Модель грузится на время вызова и сразу освобождается."""
    from faster_whisper import WhisperModel
    try:
        model = WhisperModel(MODEL, device="cuda", compute_type="int8_float16")
    except Exception:                                   # видеокарта занята — на процессоре (медленно)
        model = WhisperModel(MODEL, device="cpu", compute_type="int8")
    segs, info = model.transcribe(str(video), word_timestamps=True, vad_filter=True, beam_size=1)
    words = []
    for s in segs:
        for w in (s.words or []):
            words.append(dict(t0=round(float(w.start), 2), t1=round(float(w.end), 2), word=w.word.strip()))
    if verbose:
        print(f"[0] речь: язык {info.language}, слов {len(words)}", flush=True)
    del model
    import gc
    gc.collect()
    return words


def classify(words, lex=None):
    """слова -> моменты [{t, kind, word}] по словарю (отрицание «не» перед словом меняет знак)."""
    lex = lex or load_lexicon()
    pos, neg = tuple(lex.get("pos", [])), tuple(lex.get("neg", []))
    out, prev = [], ""
    for w in words:
        x = re.sub(r"[^\w]", "", w["word"].lower().replace("ё", "е"))
        kind = None
        if lex.get("mat_neg", True) and MAT.match(x):
            kind = "neg"
        elif x and any(x.startswith(p.replace("ё", "е")) and (len(p) > 2 or x == p) for p in neg):
            kind = "neg"
        elif x and any(x.startswith(p.replace("ё", "е")) and (len(p) > 2 or x == p) for p in pos):
            kind = "pos"
        if kind and prev in ("не", "ни", "not", "no"):
            kind = "neg" if kind == "pos" else "pos"
        if kind:
            out.append(dict(t=w["t0"], kind=kind, word=w["word"].strip(" ,.!?…\"«»").lower()))
        prev = x
    return out


def meaning_events(video, use_cache=True, verbose=True):
    video = Path(video)
    cf = CACHE / f"{video.stem[:40].strip(' .')}.words.json"
    words = None
    if use_cache and cf.exists():
        j = json.loads(cf.read_text(encoding="utf8"))
        if j.get("size") == video.stat().st_size:
            words = j["words"]
    if words is None:
        words = transcribe(video, verbose)
        CACHE.mkdir(exist_ok=True)
        cf.write_text(json.dumps(dict(size=video.stat().st_size, words=words), ensure_ascii=False), encoding="utf8")
    ev = classify(words)                                # словарь применяется каждый раз — его можно править
    if verbose:
        print(f"[0] по словарю: хорошо {sum(e['kind'] == 'pos' for e in ev)}, плохо {sum(e['kind'] == 'neg' for e in ev)}"
              f" (из {len(words)} слов)", flush=True)
    return ev


if __name__ == "__main__":
    for e in meaning_events(sys.argv[1]):
        print(f"{e['t']:7.2f}  {'+' if e['kind'] == 'pos' else '-'}  {e['word']}")
