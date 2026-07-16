import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock


INFER_SOURCE = Path(__file__).parents[1] / "monai-label" / "monailabel" / "endpoints" / "infer.py"


def load_send_response():
    source = INFER_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(INFER_SOURCE))
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "send_response"
    )
    namespace = {"os": os, "remove_file": object()}
    isolated = ast.Module(body=[function], type_ignores=[])
    exec(compile(isolated, str(INFER_SOURCE), "exec"), namespace)
    return namespace["send_response"], namespace["remove_file"]


class JsonResponseCleanupTest(unittest.TestCase):
    def test_existing_result_file_is_scheduled_for_cleanup(self):
        send_response, remove_file = load_send_response()
        background_tasks = Mock()
        datastore = Mock()

        with tempfile.NamedTemporaryFile() as result_file:
            params = {"status": "ok"}
            response = send_response(
                datastore,
                {"file": result_file.name, "params": params},
                "json",
                background_tasks,
            )

            self.assertIs(response, params)
            background_tasks.add_task.assert_called_once_with(remove_file, result_file.name)
            datastore.get_label_uri.assert_not_called()

    def test_missing_label_id_returns_json_without_datastore_lookup(self):
        send_response, _ = load_send_response()
        background_tasks = Mock()
        datastore = Mock()
        params = {"status": "ok"}

        response = send_response(
            datastore,
            {"file": "missing-label-id", "params": params},
            "json",
            background_tasks,
        )

        self.assertIs(response, params)
        background_tasks.add_task.assert_not_called()
        datastore.get_label_uri.assert_not_called()


if __name__ == "__main__":
    unittest.main()
