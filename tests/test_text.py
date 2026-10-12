import numpy as np
import pandas as pd

from airalerts import text


def stamp(clock: str) -> pd.Timestamp:
    return pd.Timestamp(f"2026-10-01 {clock}", tz="UTC")


def make_posts(rows: list[tuple[str, str, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(1, len(rows) + 1),
            "published_utc": pd.DatetimeIndex([stamp(r[0]) for r in rows]).as_unit("s"),
            "text": [r[1] for r in rows],
            "edited": [r[2] for r in rows],
        }
    )


POSTS = make_posts(
    [
        ("06:00:00", "БпЛА на Сумщині", False),
        ("11:10:00", "Реактивний БпЛА в напрямку Києва з півночі", False),
        ("11:40:00", "Пуск ракет з Ту-95", False),
        ("11:50:00", "Київ: БпЛА над містом", True),  # edited: timestamp counts, text does not
        ("12:00:00", "БпЛА курсом на столицю", False),  # published exactly at the origin: not yet known
        ("12:05:00", "Київщина: ще одна група", False),
    ]
)


def test_counts_use_only_posts_published_before_the_origin():
    row = text.build_text_features(POSTS, pd.DatetimeIndex([stamp("12:00:00")])).iloc[0]
    assert row["posts_any_60m"] == 3 and row["posts_any_180m"] == 3
    assert row["posts_kyiv_15m"] == 0 and row["posts_kyiv_60m"] == 1
    assert row["posts_missile_60m"] == 1
    assert row[text.SINCE_KYIV] == 50


def test_text_features_do_not_change_when_later_posts_are_hidden():
    rng = np.random.default_rng(0)
    seconds = np.sort(rng.integers(0, 5 * 86400, 600))
    words = np.array(["БпЛА на Київ", "КАБи на Донеччину", "Пуск ракет", "БпЛА на Одещину"])
    posts = pd.DataFrame(
        {
            "id": range(600),
            "published_utc": (stamp("00:00:00") + pd.to_timedelta(seconds, unit="s")).as_unit("s"),
            "text": words[rng.integers(0, 4, 600)],
            "edited": rng.random(600) < 0.05,
        }
    )
    offsets = rng.integers(86400, 5 * 86400, 80)
    origins = pd.DatetimeIndex([stamp("00:00:00") + pd.Timedelta(seconds=int(s)) for s in offsets]).as_unit("s")
    full = text.build_text_features(posts, origins)
    for t in origins:
        hidden = text.build_text_features(posts[posts["published_utc"] < t], pd.DatetimeIndex([t]))
        assert np.allclose(full.loc[[t]].to_numpy(dtype=float), hidden.to_numpy(dtype=float), equal_nan=True)


def test_origins_before_the_scrape_reaches_back_are_blank():
    features = text.build_text_features(POSTS, pd.DatetimeIndex([stamp("08:00:00"), stamp("12:00:00")]))
    assert features.iloc[0].isna().all() and features.iloc[1].notna().all()
