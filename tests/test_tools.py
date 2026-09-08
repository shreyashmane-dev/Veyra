import tempfile
import unittest
from pathlib import Path

from veyra.security.permissions import PermissionLevel, SecurityGuard
from veyra.tools.registry import Tool, ToolRegistry


class TestToolsAndSecurity(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = SecurityGuard(auto_approve_safe=True)
        self.registry = ToolRegistry(self.guard)

    def test_safe_calculate_tool(self) -> None:
        res = self.registry.execute("calculate", {"expression": "2 + 3 * 4"})
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["result"], "14")

    def test_caution_write_file_denied_without_callback(self) -> None:
        # Default guard blocks CAUTION tools if no approval callback is provided
        res = self.registry.execute("write_file", {"path": "dummy.txt", "content": "hello"})
        self.assertEqual(res["status"], "denied")

    def test_caution_write_file_approved_with_callback(self) -> None:
        approved_guard = SecurityGuard(
            approval_callback=lambda tool, level, args: True,
            auto_approve_safe=True,
        )
        approved_reg = ToolRegistry(approved_guard)

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "out.txt"
            res = approved_reg.execute("write_file", {"path": str(target), "content": "VEYRA active"})
            self.assertEqual(res["status"], "success")
            self.assertTrue(target.exists())
            self.assertEqual(target.read_text(), "VEYRA active")

    def test_list_tool_schemas(self) -> None:
        schemas = self.registry.list_tool_schemas()
        names = [s["name"] for s in schemas]
        self.assertIn("read_file", names)
        self.assertIn("calculate", names)
        self.assertIn("write_file", names)


if __name__ == "__main__":
    unittest.main()
