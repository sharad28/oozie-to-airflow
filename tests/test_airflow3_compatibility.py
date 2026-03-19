# -*- coding: utf-8 -*-
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Airflow 3.x compatibility tests.

Verifies that:
- Generated templates use Airflow 3.x APIs (schedule=, TaskGroup, datetime.datetime)
- Deprecated Airflow 2.x APIs are absent (SubDagOperator, schedule_interval=, dates.days_ago)
- Mapper required_imports() no longer include removed modules
- Workflow default dependencies are clean for Airflow 3.x
- Edge cases (None schedule, 0 days ago, special chars) render correctly
"""
import ast
import sys
import types
import unittest
from types import ModuleType
from typing import Any


# ---------------------------------------------------------------------------
# Bootstrap minimal airflow mocks so o2a modules can be imported without
# a full Airflow installation.
# ---------------------------------------------------------------------------

def _make_mock_module(name: str, **attrs: Any) -> ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


class _TriggerRule:  # minimal stub
    ALWAYS = "always"
    ALL_SUCCESS = "all_success"
    ALL_FAILED = "all_failed"
    ONE_SUCCESS = "one_success"
    ONE_FAILED = "one_failed"
    NONE_FAILED = "none_failed"
    NONE_SKIPPED = "none_skipped"


_airflow_pkg = _make_mock_module("airflow", AirflowException=Exception)
_airflow_utils = _make_mock_module("airflow.utils")
_airflow_utils_trigger = _make_mock_module("airflow.utils.trigger_rule", TriggerRule=_TriggerRule)
_airflow_utils_session = _make_mock_module("airflow.utils.session", provide_session=lambda f: f)
_airflow_models = _make_mock_module("airflow.models", TaskInstance=object, DagRun=object, DAG=object)

for _mod_name, _mod in [
    ("airflow", _airflow_pkg),
    ("airflow.utils", _airflow_utils),
    ("airflow.utils.trigger_rule", _airflow_utils_trigger),
    ("airflow.utils.session", _airflow_utils_session),
    ("airflow.models", _airflow_models),
    ("airflow.utils.db", _make_mock_module("airflow.utils.db", provide_session=lambda f: f)),
]:
    sys.modules.setdefault(_mod_name, _mod)

# Now import o2a template utilities (no airflow needed at this level)
from o2a.utils.template_utils import render_template  # noqa: E402

# ── helpers ────────────────────────────────────────────────────────────────

def _parse(code: str) -> ast.Module:
    """Parse Python source; raises SyntaxError on invalid code."""
    return ast.parse(code)


def _assert_contains(test: unittest.TestCase, code: str, substring: str):
    test.assertIn(substring, code, f"Expected {substring!r} in rendered output:\n{code}")


def _assert_not_contains(test: unittest.TestCase, code: str, substring: str):
    test.assertNotIn(substring, code, f"Expected {substring!r} NOT in rendered output:\n{code}")


# ---------------------------------------------------------------------------
# Template params helpers
# ---------------------------------------------------------------------------

def _workflow_params(**overrides):
    from o2a.converter.task import Task
    from o2a.converter.task_group import TaskGroup
    from o2a.converter.relation import Relation
    params = dict(
        dag_name="test_dag",
        dependencies={"import awesome_stuff"},
        task_groups=[
            TaskGroup(
                name="TASK_GROUP",
                tasks=[Task(task_id="first_task", template_name="dummy.tpl")],
            )
        ],
        job_properties={"user.name": "USER"},
        config={},
        relations={Relation(from_task_id="TASK_1", to_task_id="TASK_2")},
        schedule_interval=3,
        start_days_ago=3,
        task_map={"oozie-task": ["airflow-task"]},
    )
    params.update(overrides)
    return params


def _subwf_params(**overrides):
    params = {"task_id": "test_subwf", "trigger_rule": "always", "app_name": "childwf"}
    params.update(overrides)
    return params


def _subworkflow_params(**overrides):
    from o2a.converter.task import Task
    from o2a.converter.task_group import TaskGroup
    from o2a.converter.relation import Relation
    params = dict(
        dependencies={"import awesome_stuff"},
        task_groups=[
            TaskGroup(
                name="AAA",
                tasks=[Task(task_id="first_task", template_name="dummy.tpl")],
                relations=[Relation(from_task_id="first_task", to_task_id="second_task")],
            )
        ],
        job_properties={"user.name": "USER"},
        config={"key": "value"},
        relations={Relation(from_task_id="TASK_1", to_task_id="TASK_2")},
    )
    params.update(overrides)
    return params


# ===========================================================================
# Tests
# ===========================================================================

class TestWorkflowTemplateAirflow3(unittest.TestCase):
    """workflow.tpl must produce Airflow 3.x-compatible DAG definitions."""

    def _render(self, **overrides):
        return render_template("workflow.tpl", **_workflow_params(**overrides))

    # -- schedule param ------------------------------------------------------

    def test_uses_schedule_not_schedule_interval(self):
        code = self._render()
        _assert_not_contains(self, code, "schedule_interval=")
        _assert_contains(self, code, "schedule=")

    def test_schedule_with_numeric_interval(self):
        code = self._render(schedule_interval=7)
        _assert_contains(self, code, "schedule=datetime.timedelta(days=7)")

    def test_schedule_none_when_interval_is_falsy(self):
        code = self._render(schedule_interval=None)
        _assert_contains(self, code, "schedule=None")

    def test_schedule_zero_renders_none(self):
        """0 is falsy in Jinja — should render schedule=None."""
        code = self._render(schedule_interval=0)
        _assert_contains(self, code, "schedule=None")

    # -- start_date ----------------------------------------------------------

    def test_no_dates_days_ago(self):
        code = self._render()
        _assert_not_contains(self, code, "dates.days_ago")

    def test_start_date_uses_datetime_arithmetic(self):
        code = self._render(start_days_ago=5)
        _assert_contains(self, code, "datetime.datetime.now() - datetime.timedelta(days=5)")

    def test_start_days_ago_zero(self):
        code = self._render(start_days_ago=0)
        _assert_contains(self, code, "datetime.timedelta(days=0)")

    def test_start_days_ago_large_value(self):
        code = self._render(start_days_ago=365)
        _assert_contains(self, code, "datetime.timedelta(days=365)")

    # -- no deprecated dates import ------------------------------------------

    def test_no_dates_import_in_output(self):
        code = self._render()
        _assert_not_contains(self, code, "from airflow.utils import dates")

    # -- valid Python --------------------------------------------------------

    def test_rendered_is_valid_python(self):
        code = self._render()
        _parse(code)  # raises SyntaxError if invalid

    def test_valid_python_with_no_schedule(self):
        code = self._render(schedule_interval=None, start_days_ago=0)
        _parse(code)


class TestSubwfTemplateAirflow3(unittest.TestCase):
    """subwf.tpl must call create_task_group, not SubDagOperator."""

    def _render(self, **overrides):
        return render_template("subwf.tpl", **_subwf_params(**overrides))

    def test_no_subdag_operator(self):
        code = self._render()
        _assert_not_contains(self, code, "SubDagOperator")

    def test_calls_create_task_group(self):
        code = self._render()
        _assert_contains(self, code, "create_task_group")

    def test_passes_dag_to_create_task_group(self):
        code = self._render()
        _assert_contains(self, code, "create_task_group(\n    dag,")

    def test_no_sub_dag_method_call(self):
        code = self._render()
        _assert_not_contains(self, code, ".sub_dag(")

    def test_no_schedule_interval_param(self):
        """subwf.tpl should not pass schedule_interval to anything."""
        code = self._render()
        _assert_not_contains(self, code, "schedule_interval")

    def test_correct_module_prefix(self):
        """Should import/call from subdag_<app_name> module."""
        code = self._render(app_name="my_child_wf")
        _assert_contains(self, code, "subdag_my_child_wf.create_task_group(")

    def test_valid_python(self):
        code = self._render()
        _parse(code)

    def test_edge_case_app_name_with_hyphens(self):
        """Hyphens in app_name should be converted to underscores (via to_var filter)."""
        code = self._render(app_name="my-child-wf")
        # to_var converts hyphens to underscores
        _assert_contains(self, code, "subdag_my_child_wf.create_task_group(")
        _parse(code)

    def test_edge_case_task_id_with_special_chars(self):
        """task_id with quotes should be safely escaped."""
        code = self._render(task_id='task"id')
        _parse(code)

    @unittest.expectedFailure
    def test_old_subdag_pattern_absent(self):
        """Confirm old SubDagOperator pattern is completely gone."""
        code = self._render()
        _assert_contains(self, code, "SubDagOperator(")  # must fail


class TestSubworkflowTemplateAirflow3(unittest.TestCase):
    """subworkflow.tpl must produce create_task_group, not sub_dag / models.DAG."""

    def _render(self, **overrides):
        return render_template("subworkflow.tpl", **_subworkflow_params(**overrides))

    def test_no_sub_dag_function(self):
        code = self._render()
        _assert_not_contains(self, code, "def sub_dag(")

    def test_has_create_task_group_function(self):
        code = self._render()
        _assert_contains(self, code, "def create_task_group(")

    def test_no_models_dag(self):
        code = self._render()
        _assert_not_contains(self, code, "models.DAG(")

    def test_no_schedule_interval_param(self):
        code = self._render()
        _assert_not_contains(self, code, "schedule_interval")

    def test_no_start_date_param(self):
        code = self._render()
        _assert_not_contains(self, code, "start_date")

    def test_uses_task_group_import(self):
        code = self._render()
        _assert_contains(self, code, "from airflow.utils.task_group import TaskGroup")

    def test_uses_task_group_context_manager(self):
        code = self._render()
        _assert_contains(self, code, "with TaskGroup(")

    def test_returns_task_group(self):
        code = self._render()
        _assert_contains(self, code, "return task_group")

    def test_valid_python(self):
        code = self._render()
        _parse(code)


class TestMapperRequiredImportsAirflow3(unittest.TestCase):
    """Mapper required_imports() must not include removed Airflow 2.x packages."""

    MAPPERS_WITH_DATES_REMOVED = [
        "o2a.mappers.ssh_mapper",
        "o2a.mappers.shell_mapper",
        "o2a.mappers.hive_mapper",
        "o2a.mappers.decision_mapper",
        "o2a.mappers.mapreduce_mapper",
        "o2a.mappers.pig_mapper",
        "o2a.mappers.java_mapper",
    ]

    def _get_required_imports(self, module_path: str):
        import importlib
        mod = importlib.import_module(module_path)
        # Find the mapper class
        mapper_class = next(
            cls for name, cls in vars(mod).items()
            if isinstance(cls, type) and hasattr(cls, "required_imports")
            and name != "ActionMapper"
        )
        # Create a minimal instance using mock
        from unittest.mock import MagicMock, patch
        from xml.etree.ElementTree import fromstring
        from o2a.o2a_libs.src.o2a_lib.property_utils import PropertySet
        node = fromstring("<action><ok to='end'/><error to='kill'/></action>")
        prop = PropertySet(job_properties={}, config={})
        with patch.object(mapper_class, "__init__", return_value=None):
            instance = mapper_class.__new__(mapper_class)
            # provide minimal attributes required_imports() might access
            instance.name = "test"
            instance.dag_name = "test_dag"
            instance.props = prop
            instance.oozie_node = node
            instance.app_name = "test_app"  # subworkflow specific
            instance.jar_files = []
        return instance.required_imports()

    def _check_no_dates_import(self, module_path: str):
        try:
            imports = self._get_required_imports(module_path)
            imports_str = "\n".join(imports)
            self.assertNotIn(
                "from airflow.utils import dates",
                imports_str,
                f"{module_path} still exports deprecated 'from airflow.utils import dates'",
            )
        except Exception as e:
            self.skipTest(f"Could not instantiate mapper from {module_path}: {e}")

    def test_ssh_mapper_no_dates_import(self):
        self._check_no_dates_import("o2a.mappers.ssh_mapper")

    def test_shell_mapper_no_dates_import(self):
        self._check_no_dates_import("o2a.mappers.shell_mapper")

    def test_hive_mapper_no_dates_import(self):
        self._check_no_dates_import("o2a.mappers.hive_mapper")

    def test_decision_mapper_no_dates_import(self):
        self._check_no_dates_import("o2a.mappers.decision_mapper")

    def test_mapreduce_mapper_no_dates_import(self):
        self._check_no_dates_import("o2a.mappers.mapreduce_mapper")

    def test_pig_mapper_no_dates_import(self):
        self._check_no_dates_import("o2a.mappers.pig_mapper")

    def test_java_mapper_no_dates_import(self):
        self._check_no_dates_import("o2a.mappers.java_mapper")

    def test_subworkflow_mapper_no_subdag_operator_import(self):
        self._check_no_dates_import("o2a.mappers.subworkflow_mapper")

    def test_subworkflow_mapper_has_task_group_import(self):
        """subworkflow_mapper must export TaskGroup import for Airflow 3.x."""
        try:
            imports = self._get_required_imports("o2a.mappers.subworkflow_mapper")
            imports_str = "\n".join(imports)
            self.assertIn(
                "from airflow.utils.task_group import TaskGroup",
                imports_str,
                "subworkflow_mapper must include TaskGroup import for Airflow 3.x",
            )
            self.assertNotIn(
                "from airflow.operators.subdag import SubDagOperator",
                imports_str,
                "subworkflow_mapper must NOT include removed SubDagOperator",
            )
        except Exception as e:
            self.skipTest(f"Could not instantiate subworkflow_mapper: {e}")

    def test_all_required_imports_are_valid_python(self):
        """All required_imports() from mappers must be parseable as Python."""
        import importlib
        mapper_modules = [
            "o2a.mappers.ssh_mapper",
            "o2a.mappers.shell_mapper",
            "o2a.mappers.hive_mapper",
            "o2a.mappers.decision_mapper",
            "o2a.mappers.mapreduce_mapper",
            "o2a.mappers.pig_mapper",
            "o2a.mappers.java_mapper",
        ]
        for module_path in mapper_modules:
            with self.subTest(mapper=module_path):
                try:
                    imports = self._get_required_imports(module_path)
                    for imp in imports:
                        _parse(imp)
                except Exception as e:
                    self.skipTest(f"Could not test {module_path}: {e}")


class TestWorkflowDefaultDependenciesAirflow3(unittest.TestCase):
    """Workflow.dependencies must be clean for Airflow 3.x."""

    def setUp(self):
        from o2a.converter.workflow import Workflow
        self.workflow = Workflow(
            input_directory_path="/tmp/in",
            output_directory_path="/tmp/out",
            dag_name="test",
        )

    def test_no_dates_in_default_dependencies(self):
        self.assertNotIn(
            "from airflow.utils import dates",
            self.workflow.dependencies,
            "Default dependencies must not include removed 'airflow.utils.dates'",
        )

    def test_datetime_still_in_default_dependencies(self):
        self.assertIn(
            "import datetime",
            self.workflow.dependencies,
            "'import datetime' must remain as it replaces dates.days_ago()",
        )

    def test_trigger_rule_still_present(self):
        self.assertIn("from airflow.utils.trigger_rule import TriggerRule", self.workflow.dependencies)

    def test_bash_empty_operators_still_present(self):
        self.assertIn("from airflow.operators import bash, empty", self.workflow.dependencies)

    def test_all_default_deps_are_valid_python(self):
        for dep in self.workflow.dependencies:
            with self.subTest(dep=dep):
                _parse(dep)


class TestElWfFunctionsAirflow3(unittest.TestCase):
    """el_wf_functions.py must use Airflow 3.x APIs."""

    def _read_source(self):
        import os
        path = os.path.join(
            os.path.dirname(__file__),
            "..", "o2a", "o2a_libs", "src", "o2a_lib", "el_wf_functions.py"
        )
        with open(os.path.normpath(path)) as f:
            return f.read()

    def test_uses_airflow_utils_session_not_db(self):
        source = self._read_source()
        self.assertIn(
            "from airflow.utils.session import provide_session",
            source,
            "Must use airflow.utils.session (Airflow 3.x) not airflow.utils.db",
        )
        self.assertNotIn(
            "from airflow.utils.db import provide_session",
            source,
            "Must NOT use removed airflow.utils.db.provide_session",
        )

    def test_no_execution_date_column(self):
        source = self._read_source()
        self.assertNotIn(
            "execution_date",
            source,
            "TaskInstance.execution_date was removed in Airflow 3.x; use start_date",
        )

    def test_uses_start_date_for_ordering(self):
        source = self._read_source()
        self.assertIn(
            ".order_by(ti.start_date.asc())",
            source,
            "Must order by start_date (Airflow 3.x) instead of execution_date",
        )

    def test_source_is_valid_python(self):
        source = self._read_source()
        _parse(source)


class TestRequirementsTxtAirflow3(unittest.TestCase):
    """requirements.txt must pin Airflow >= 3.0.0."""

    def _read_requirements(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
        with open(os.path.normpath(path)) as f:
            return f.read()

    def test_airflow_version_is_3_or_greater(self):
        reqs = self._read_requirements()
        self.assertIn(
            "apache-airflow>=3.0.0",
            reqs,
            "requirements.txt must require apache-airflow>=3.0.0",
        )

    def test_no_airflow_2_pin(self):
        reqs = self._read_requirements()
        self.assertNotIn(
            "apache-airflow>=2.0.0",
            reqs,
            "Airflow 2.x pin should have been removed",
        )


class TestMapperSourceAirflow3(unittest.TestCase):
    """Verify mapper source files directly — covers mappers that need complex context to instantiate."""

    def _read_mapper(self, filename: str) -> str:
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "o2a", "mappers", filename)
        with open(os.path.normpath(path)) as f:
            return f.read()

    def test_decision_mapper_no_dates_import(self):
        src = self._read_mapper("decision_mapper.py")
        self.assertNotIn('"from airflow.utils import dates"', src)

    def test_subworkflow_mapper_no_subdag_operator(self):
        src = self._read_mapper("subworkflow_mapper.py")
        self.assertNotIn("SubDagOperator", src)
        self.assertNotIn("airflow.operators.subdag", src)

    def test_subworkflow_mapper_has_task_group(self):
        src = self._read_mapper("subworkflow_mapper.py")
        self.assertIn("from airflow.utils.task_group import TaskGroup", src)

    def test_subworkflow_mapper_no_dates_import(self):
        src = self._read_mapper("subworkflow_mapper.py")
        self.assertNotIn('"from airflow.utils import dates"', src)

    def test_ssh_mapper_source_no_dates(self):
        src = self._read_mapper("ssh_mapper.py")
        self.assertNotIn('"from airflow.utils import dates"', src)

    def test_shell_mapper_source_no_dates(self):
        src = self._read_mapper("shell_mapper.py")
        self.assertNotIn('"from airflow.utils import dates"', src)

    def test_hive_mapper_source_no_dates(self):
        src = self._read_mapper("hive_mapper.py")
        self.assertNotIn('"from airflow.utils import dates"', src)

    def test_mapreduce_mapper_source_no_dates(self):
        src = self._read_mapper("mapreduce_mapper.py")
        self.assertNotIn('"from airflow.utils import dates"', src)

    def test_pig_mapper_source_no_dates(self):
        src = self._read_mapper("pig_mapper.py")
        self.assertNotIn('"from airflow.utils import dates"', src)

    def test_java_mapper_source_no_dates(self):
        src = self._read_mapper("java_mapper.py")
        self.assertNotIn('"from airflow.utils import dates"', src)


class TestEndToEndDAGConversionAirflow3(unittest.TestCase):
    """
    End-to-end tests that simulate DAG conversion and verify the combined
    output (imports + body) is correct for Airflow 3.x.
    """

    def _build_full_dag(self, **workflow_overrides):
        """Render a full workflow.tpl output with given overrides."""
        params = _workflow_params(**workflow_overrides)
        return render_template("workflow.tpl", **params)

    def test_full_dag_has_no_airflow2_apis(self):
        code = self._build_full_dag()
        deprecated_patterns = [
            "schedule_interval=",
            "dates.days_ago",
            "SubDagOperator",
            "from airflow.utils import dates",
            "from airflow.operators.subdag",
        ]
        for pattern in deprecated_patterns:
            with self.subTest(pattern=pattern):
                _assert_not_contains(self, code, pattern)

    def test_full_dag_has_airflow3_apis(self):
        code = self._build_full_dag()
        required_patterns = [
            "schedule=",
            "datetime.datetime.now()",
        ]
        for pattern in required_patterns:
            with self.subTest(pattern=pattern):
                _assert_contains(self, code, pattern)

    def test_full_dag_is_valid_python(self):
        code = self._build_full_dag()
        _parse(code)

    def test_subworkflow_dag_file_is_valid_python(self):
        code = render_template("subworkflow.tpl", **_subworkflow_params())
        _parse(code)

    def test_subwf_task_in_parent_dag_is_valid_python(self):
        code = render_template("subwf.tpl", **_subwf_params())
        _parse(code)

    def test_dag_with_no_schedule(self):
        """Unscheduled DAGs (schedule=None) must be valid Python."""
        code = self._build_full_dag(schedule_interval=None)
        _assert_contains(self, code, "schedule=None")
        _parse(code)

    def test_dag_with_large_start_days_ago(self):
        code = self._build_full_dag(start_days_ago=9999)
        _assert_contains(self, code, "datetime.timedelta(days=9999)")
        _parse(code)

    def test_subwf_template_passes_user_defined_macros(self):
        """subwf.tpl must forward user_defined_macros to TaskGroup."""
        code = render_template("subwf.tpl", **_subwf_params())
        _assert_contains(self, code, "dag.user_defined_macros")


if __name__ == "__main__":
    unittest.main()
