"""SQLite database for training presets and caption presets."""

import json
import os
import sqlite3
import time
from typing import Optional


class PresetsDB:
    """Manages training presets and caption presets in SQLite."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to a db file next to this module
            db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "digit_presets.db"
            )
        self.db_path = os.path.abspath(db_path)
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS training_presets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT DEFAULT '',
                    model_type TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS caption_presets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT DEFAULT '',
                    system_prompt TEXT NOT NULL,
                    prompt_template TEXT NOT NULL,
                    model TEXT DEFAULT 'gemini-2.5-flash',
                    temperature REAL DEFAULT 0.4,
                    max_tokens INTEGER DEFAULT 300,
                    example_captions TEXT DEFAULT '[]',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    preset_name TEXT,
                    config_json TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    started_at REAL,
                    completed_at REAL,
                    current_step INTEGER DEFAULT 0,
                    total_steps INTEGER DEFAULT 0,
                    loss REAL,
                    output_path TEXT,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS naming_presets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT DEFAULT '',
                    output_dir_template TEXT NOT NULL,
                    lora_name_template TEXT NOT NULL,
                    checkpoint_template TEXT DEFAULT 'step_{step:06d}',
                    sample_template TEXT DEFAULT 'sample_{prompt_index:02d}_{step:06d}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trigger_presets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT DEFAULT '',
                    trigger_word TEXT NOT NULL,
                    trigger_class TEXT DEFAULT '',
                    trigger_phrase TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sample_prompt_presets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT DEFAULT '',
                    prompts_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- Training Presets ---

    def save_training_preset(self, name: str, model_type: str,
                              config: dict, description: str = "") -> int:
        now = time.time()
        with self._connect() as conn:
            # Upsert
            existing = conn.execute(
                "SELECT id FROM training_presets WHERE name = ?", (name,)
            ).fetchone()
            if existing:
                conn.execute("""
                    UPDATE training_presets
                    SET description = ?, model_type = ?, config_json = ?, updated_at = ?
                    WHERE name = ?
                """, (description, model_type, json.dumps(config), now, name))
                return existing["id"]
            else:
                cursor = conn.execute("""
                    INSERT INTO training_presets
                    (name, description, model_type, config_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (name, description, model_type, json.dumps(config), now, now))
                return cursor.lastrowid

    def get_training_preset(self, name: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM training_presets WHERE name = ?", (name,)
            ).fetchone()
            if row:
                result = dict(row)
                result["config"] = json.loads(result.pop("config_json"))
                return result
            return None

    def list_training_presets(self, model_type: str = None) -> list:
        with self._connect() as conn:
            if model_type:
                rows = conn.execute(
                    "SELECT name, description, model_type, updated_at "
                    "FROM training_presets WHERE model_type = ? ORDER BY updated_at DESC",
                    (model_type,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT name, description, model_type, updated_at "
                    "FROM training_presets ORDER BY updated_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def delete_training_preset(self, name: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM training_presets WHERE name = ?", (name,)
            )
            return cursor.rowcount > 0

    # --- Caption Presets ---

    def save_caption_preset(self, name: str, system_prompt: str,
                             prompt_template: str, model: str = "gemini-2.5-flash",
                             temperature: float = 0.4, max_tokens: int = 300,
                             example_captions: list = None,
                             description: str = "") -> int:
        now = time.time()
        examples_json = json.dumps(example_captions or [])
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM caption_presets WHERE name = ?", (name,)
            ).fetchone()
            if existing:
                conn.execute("""
                    UPDATE caption_presets
                    SET description = ?, system_prompt = ?, prompt_template = ?,
                        model = ?, temperature = ?, max_tokens = ?,
                        example_captions = ?, updated_at = ?
                    WHERE name = ?
                """, (description, system_prompt, prompt_template,
                      model, temperature, max_tokens, examples_json, now, name))
                return existing["id"]
            else:
                cursor = conn.execute("""
                    INSERT INTO caption_presets
                    (name, description, system_prompt, prompt_template,
                     model, temperature, max_tokens, example_captions,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (name, description, system_prompt, prompt_template,
                      model, temperature, max_tokens, examples_json, now, now))
                return cursor.lastrowid

    def get_caption_preset(self, name: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM caption_presets WHERE name = ?", (name,)
            ).fetchone()
            if row:
                result = dict(row)
                result["example_captions"] = json.loads(result["example_captions"])
                return result
            return None

    def list_caption_presets(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, description, model, updated_at "
                "FROM caption_presets ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_caption_preset(self, name: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM caption_presets WHERE name = ?", (name,)
            )
            return cursor.rowcount > 0

    # --- Naming Convention Presets ---

    def save_naming_preset(self, name: str, output_dir_template: str,
                            lora_name_template: str,
                            checkpoint_template: str = "step_{step:06d}",
                            sample_template: str = "sample_{prompt_index:02d}_{step:06d}",
                            description: str = "") -> int:
        now = time.time()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM naming_presets WHERE name = ?", (name,)
            ).fetchone()
            if existing:
                conn.execute("""
                    UPDATE naming_presets
                    SET description = ?, output_dir_template = ?,
                        lora_name_template = ?, checkpoint_template = ?,
                        sample_template = ?, updated_at = ?
                    WHERE name = ?
                """, (description, output_dir_template, lora_name_template,
                      checkpoint_template, sample_template, now, name))
                return existing["id"]
            else:
                cursor = conn.execute("""
                    INSERT INTO naming_presets
                    (name, description, output_dir_template, lora_name_template,
                     checkpoint_template, sample_template, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (name, description, output_dir_template, lora_name_template,
                      checkpoint_template, sample_template, now, now))
                return cursor.lastrowid

    def get_naming_preset(self, name: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM naming_presets WHERE name = ?", (name,)
            ).fetchone()
            return dict(row) if row else None

    def list_naming_presets(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, description, output_dir_template, lora_name_template, updated_at "
                "FROM naming_presets ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_naming_preset(self, name: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM naming_presets WHERE name = ?", (name,)
            )
            return cursor.rowcount > 0

    # --- Trigger Word Presets ---

    def save_trigger_preset(self, name: str, trigger_word: str,
                             trigger_class: str = "",
                             trigger_phrase: str = "",
                             description: str = "") -> int:
        now = time.time()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM trigger_presets WHERE name = ?", (name,)
            ).fetchone()
            if existing:
                conn.execute("""
                    UPDATE trigger_presets
                    SET description = ?, trigger_word = ?, trigger_class = ?,
                        trigger_phrase = ?, updated_at = ?
                    WHERE name = ?
                """, (description, trigger_word, trigger_class, trigger_phrase, now, name))
                return existing["id"]
            else:
                cursor = conn.execute("""
                    INSERT INTO trigger_presets
                    (name, description, trigger_word, trigger_class,
                     trigger_phrase, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (name, description, trigger_word, trigger_class,
                      trigger_phrase, now, now))
                return cursor.lastrowid

    def get_trigger_preset(self, name: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trigger_presets WHERE name = ?", (name,)
            ).fetchone()
            return dict(row) if row else None

    def list_trigger_presets(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, description, trigger_word, trigger_class, updated_at "
                "FROM trigger_presets ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_trigger_preset(self, name: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM trigger_presets WHERE name = ?", (name,)
            )
            return cursor.rowcount > 0

    # --- Sample Prompt Presets ---

    def save_sample_prompt_preset(self, name: str, prompts: list,
                                    description: str = "") -> int:
        """Save a set of sample prompts as a preset.

        Prompts should contain [trigger], [trigger_class], [trigger_phrase]
        placeholders that get resolved at training time.
        """
        now = time.time()
        prompts_json = json.dumps(prompts)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM sample_prompt_presets WHERE name = ?", (name,)
            ).fetchone()
            if existing:
                conn.execute("""
                    UPDATE sample_prompt_presets
                    SET description = ?, prompts_json = ?, updated_at = ?
                    WHERE name = ?
                """, (description, prompts_json, now, name))
                return existing["id"]
            else:
                cursor = conn.execute("""
                    INSERT INTO sample_prompt_presets
                    (name, description, prompts_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, description, prompts_json, now, now))
                return cursor.lastrowid

    def get_sample_prompt_preset(self, name: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sample_prompt_presets WHERE name = ?", (name,)
            ).fetchone()
            if row:
                result = dict(row)
                result["prompts"] = json.loads(result.pop("prompts_json"))
                return result
            return None

    def list_sample_prompt_presets(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, description, updated_at "
                "FROM sample_prompt_presets ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_sample_prompt_preset(self, name: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sample_prompt_presets WHERE name = ?", (name,)
            )
            return cursor.rowcount > 0

    # --- Training Runs (history) ---

    def create_run(self, name: str, config: dict,
                    preset_name: str = None, total_steps: int = 0) -> int:
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO training_runs
                (name, preset_name, config_json, status, total_steps)
                VALUES (?, ?, ?, 'pending', ?)
            """, (name, preset_name, json.dumps(config), total_steps))
            return cursor.lastrowid

    def update_run(self, run_id: int, **kwargs):
        allowed = {
            "status", "started_at", "completed_at",
            "current_step", "loss", "output_path", "error_message",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [run_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE training_runs SET {set_clause} WHERE id = ?", values
            )

    def get_run(self, run_id: int) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row:
                result = dict(row)
                result["config"] = json.loads(result.pop("config_json"))
                return result
            return None

    def list_runs(self, limit: int = 20) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, preset_name, status, current_step, "
                "total_steps, loss, started_at, completed_at "
                "FROM training_runs ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
