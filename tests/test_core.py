import shutil
import tempfile
import unittest
from pathlib import Path

from veyra.core.cognitive import CognitiveCore
from veyra.core.state import CognitiveState
from veyra.memory.database import MemoryDB
from veyra.model.starter import StarterLanguageEngine


class TestCognitiveCore(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db = MemoryDB(self.temp_dir / "memory.db")
        self.engine = StarterLanguageEngine(self.db)
        self.core = CognitiveCore(self.engine, self.db, CognitiveState())

    def tearDown(self) -> None:
        self.db.close()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_teach_and_recall(self) -> None:
        self.assertIn("Learned", self.core.handle("teach CN is Computer Networks"))
        self.assertIn("Computer Networks", self.core.handle("what do you remember"))

    def test_identity(self) -> None:
        self.assertIn("VEYRA", self.core.handle("who are you"))

    def test_commands(self) -> None:
        self.assertIn("Commands:", self.core.handle("/help"))
        self.assertIn("Engine:", self.core.handle("/model"))
        self.assertIn("Available tools", self.core.handle("/tools"))


if __name__ == "__main__":
    unittest.main()
