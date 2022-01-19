import os
import re
import peewee
import textwrap
import decimal
import enum
import contextlib
from datetime import datetime, date, time
from collections import OrderedDict, namedtuple

try:
    from playhouse.postgres_ext import (
        ArrayField as PgArrayField,
        HStoreField as PgHStoreField,
        BinaryJSONField as PgBinaryJSONField,
        TSVectorField as PgTSVectorField,
    )
    import psycopg2

except ImportError:
    PgArrayField = PgHStoreField = PgBinaryJSONField = PgTSVectorField = None


__all__ = ('Router', 'Snapshot', 'Migrator', 'MigrationError', 'deconstructor')


INDENT = ' ' * 4
NEWLINE = '\n'
MIGRATE_TEMPLATE = """# auto-generated snapshot
from peewee import *
{imports}


snapshot = Snapshot()

{snapshot}

"""


class MigrationError(Exception):
    pass


class LiteralBlock:

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.value == other.value

    def __repr__(self):
        return self.value


class CallBlock:

    def __init__(self, fn, args=None, kwargs=None):
        self.fn = fn
        self.args = args or ()
        self.kwargs = kwargs or {}

    def __eq__(self, other):
        return (self.__class__ is other.__class__
                and self.fn == other.fn
                and self.args == other.args
                and self.kwargs == other.kwargs)

    def __repr__(self):
        params = []
        if self.args:
            params += [repr(p) for p in self.args]
        if self.kwargs:
            params += ['%s=%r' % (k, v) for k, v in self.kwargs.items()]
        return '%s(%s)' % (self.fn, ', '.join(params))


def deconstructor(field_class):
    def decorator(fn):
        Column._deconstructors[field_class] = fn
        return fn
    return decorator


def get_constraints(constraints):
    result = []
    for c in constraints:
        if not isinstance(c, peewee.SQL):
            raise TypeError('Constraint must be SQL object.')
        args = (c.sql, c.params) if c.params else (c.sql,)
        result.append(CallBlock('SQL', args))
    return result


def value_repr(value):
    if isinstance(value, enum.Enum):
        return repr(value.value)
    return repr(value)


class Column:

    __slots__ = ('modules', 'field_class', 'name', '__dict__')

    _default_callables = {
        datetime.now: ('datetime', 'datetime.now'),
        datetime.utcnow: ('datetime', 'datetime.utcnow'),
    }

    _deconstructors = {}  # type: dict

    def __init__(self, field, complete=False):
        self.modules = set()
        self.field_class = getattr(field, 'deconstruct_as', type(field))
        self.name = field.name
        self.__dict__ = self._deconstruct_field(field, complete)
        self.modules.add(self.field_class.__module__)

    def __eq__(self, other):
        return (
            self.__class__ is other.__class__
            and self.field_class is other.field_class
            and self.__dict__ == other.__dict__
            )

    def to_code(self):
        params = self.__dict__
        param_str = ', '.join('%s=%s' % (k, value_repr(v)) for k, v in sorted(params.items()))
        if issubclass(self.field_class, peewee.ForeignKeyField):
            name = 'snapshot.ForeignKeyField'
        else:
            module, name = self.field_class.__module__, self.field_class.__name__
            if module != 'peewee' or name not in peewee.__all__:
                name = module + '.' + name
        result = '%s = %s(%s)' % (
            self.name,
            name,
            param_str)
        return result

    def to_params(self, exclude=()):
        params = dict(self.__dict__)
        if exclude:
            for key in exclude:
                params.pop(key, None)
        return params

    def _deconstruct_field(self, field, complete):
        default_args = (
            ('null', False),
            ('index', False),
            ('unique', False),
            ('primary_key', False),
            ('constraints', None),
            ('sequence', None),
            ('collation', None),
            ('unindexed', False),
        )

        params = {}

        for attr, value in default_args:
            if complete or getattr(field, attr) != value:
                params[attr] = getattr(field, attr)

        if complete or field.column_name != field.name:
            params['column_name'] = field.column_name

        # Handle extra attributes
        if hasattr(field, 'deconstruct'):
            field.deconstruct(params)
        else:
            for cls in type(field).__mro__:
                if cls in self._deconstructors:
                    self._deconstructors[cls](field, params=params, complete=complete)
                    break

        # Handle default value.
        if field.default is not None:
            params['default'] = field.default

        for k in params:
            param = params[k]
            if callable(param):
                if param in self._default_callables:
                    path = self._default_callables[param]
                else:
                    path = param.__module__, param.__name__
                if None in path:
                    raise TypeError("Can't use %r as field argument." % param)
                self.modules.add(path[0])
                params[k] = LiteralBlock('%s.%s' % path)

        if field.constraints:
            params['constraints'] = get_constraints(field.constraints)

        return params


@deconstructor(peewee.DateTimeField)
def datetimefield_deconstructor(field, params, **_):
    if not isinstance(field.formats, list):
        params['formats'] = field.formats


@deconstructor(peewee.CharField)
def charfield_deconstructor(field, params, **_):
    params['max_length'] = field.max_length


@deconstructor(peewee.DecimalField)
def decimalfield_deconstructor(field, params, **_):
    params['max_digits'] = field.max_digits
    params['decimal_places'] = field.decimal_places
    params['auto_round'] = field.auto_round
    params['rounding'] = field.rounding


if PgArrayField:
    @deconstructor(PgArrayField)
    def arrayfield_deconstructor(field, params, **_):
        params['dimensions'] = field.dimensions
        params['convert_values'] = field.convert_values
        params['field_class'] = field._ArrayField__field.__class__


if PgHStoreField:
    @deconstructor(PgHStoreField)
    def hstorefield_deconstructor(field, params, **_):
        params['index'] = field.index


if PgBinaryJSONField:
    @deconstructor(PgBinaryJSONField)
    def binaryjsonfield_deconstructor(field, params, **_):
        params['index'] = field.index


if PgTSVectorField:
    @deconstructor(PgTSVectorField)
    def tsvectorfield_deconstructor(field, params, **_):
        params['index'] = field.index


@deconstructor(peewee.ForeignKeyField)
def deconstruct_foreignkey(field, params, complete):
    default_column_name = field.name
    if not default_column_name.endswith('_id'):
        default_column_name += '_id'
    if complete or default_column_name != field.column_name:
        params['column_name'] = field.column_name
    else:
        params.pop('column_name', None)
    if complete or field.rel_field.name != field.rel_model._meta.primary_key.name:
        params['field'] = field.rel_field.name
    if complete or (field.backref and field.backref != field.model._meta.name + '_set'):
        params['backref'] = field.backref
    default_object_id_name = field.column_name
    if default_object_id_name == field.name:
        default_object_id_name += '_id'
    if complete or default_object_id_name != field.object_id_name:
        params['object_id_name'] = field.object_id_name
    if field.on_delete:
        params['on_delete'] = field.on_delete
    if field.on_update:
        params['on_update'] = field.on_update
    if field.deferrable:
        params['deferrable'] = field.deferrable
    if field.model is field.rel_model:
        params['model'] = '@self'
    else:
        params['model'] = field.rel_model._meta.name


@deconstructor(peewee.DeferredForeignKey)
def deconstruct_deferredforeignkey(field, **_):
    raise TypeError("DeferredForeignKey '%s.%s' will not be resolved, use ForeignKeyField instead." % (
        field.model.__name__, field.name))


class Orm:
    def __init__(self, models):
        self._items = list(models)
        self._mapping = {
            model._meta.name: model for model in models
        }

    def __iter__(self):
        return iter(self._items)

    def get(self, item):
        return self._mapping.get(item.lower())

    def __getitem__(self, item):
        return self._mapping[item.lower()]

    def __getattr__(self, item):
        try:
            return self._mapping[item.lower()]
        except KeyError:
            raise AttributeError(item)


class Snapshot:

    def __init__(self, database, models):
        self.mapping = {}
        self.items = []
        self.models = models
        self.database = database

    def __getitem__(self, name):
        return self.mapping[name]

    def __iter__(self):
        return iter(self.items)

    def __repr__(self):
        return repr(self.items)

    def get_orm(self):
        return Orm(self.items)

    def append(self, model):
        model._meta.database = self.database
        self.items.append(model)
        self.mapping[model._meta.name] = model
        return model

    def ForeignKeyField(self, model, **kwargs):
        if model == '@self':
            return peewee.ForeignKeyField(model='self', **kwargs)
        if model in self.mapping:
            return peewee.ForeignKeyField(self.mapping[model], **kwargs)
        else:
            for m in self.models:
                if m._meta.name == model:
                    break
            else:
                raise TypeError('Model "%s" is used, but not included to the watch list.' % model)
            return peewee.DeferredForeignKey(model, **kwargs)


class cached_property:

    def __init__(self, func, name=None):
        self.func = func
        self.name = name or func.__name__

    def __get__(self, instance, cls=None):
        if instance is None:
            return self
        res = instance.__dict__[self.name] = self.func(instance)
        return res


class NodeDeconstructor:

    def get_code(self, node):
        for tp in type(node).__mro__:
            meth = getattr(self, 'code_' + tp.__name__, None)
            if meth is not None:
                return meth(node)
        else:
            self.generic_code(node)

    def generic_code(self, node):
        raise ValueError('Unsupported node: %r.' % node)

    @staticmethod
    def code_SQL(node):
        return CallBlock('SQL', args=(node.sql, node.params))

    @staticmethod
    def code_Table(node):
        kwargs = {'name': node.__name__}
        if node._schema:
            kwargs['schema'] = node._schema
        return CallBlock('Table', kwargs=kwargs)

    def code_Index(self, node):
        kwargs = {
            'name': node._name,
            'table': self.get_code(node._table),
            'expressions': [self.get_code(x) for x in node._expressions],
        }
        if node._unique:
            kwargs['unique'] = True
        if node._safe:
            kwargs['safe'] = True
        if node._using:
            kwargs['using'] = node._using
        assert node._where is None
        return CallBlock('Index', kwargs=kwargs)

    def code_Expression(self, node):
        args = (
            self.get_code(node.lhs),
            node.op,
            self.get_code(node.rhs),
            )
        kwargs = {}
        if node.flat:
            kwargs['flat'] = True
        return CallBlock('peewee.Expression', args=args, kwargs=kwargs)

    def code_Field(self, node):
        return self.get_code(node.column)

    def code_Column(self, node):
        return CallBlock('Column', args=(self.get_code(node.source), node.name))


class Compiler:

    def __init__(self, database, models, modules=('datetime', 'peewee')):
        self.database = database
        self.modules = set(modules)
        self.models = models
        self.code = []

    @cached_property
    def snapshot(self):
        models = peewee.sort_models(self.models)
        snapshot = []
        for model in models:
            snapshot.append("@snapshot.append\n" + self.model_to_code(model))
        return snapshot

    @cached_property
    def module_code(self):
        snapshot = NEWLINE + (NEWLINE + NEWLINE).join(self.snapshot)
        imports = NEWLINE.join(['import %s' % x for x in sorted(self.modules)])
        return MIGRATE_TEMPLATE.format(snapshot=snapshot, imports=imports)

    def add_code(self, code):
        self.code.append(code)

    def get_code(self):
        code = self.module_code
        if self.code:
            code += '\n\n'.join(self.code)
        return code

    def model_to_code(self, model):
        template = (
            "class {classname}(peewee.Model):\n"
            + INDENT + "{fields}\n"
            + INDENT + "{meta}\n"
        )

        field_code = []
        for field in model._meta.sorted_fields:
            if type(field) is peewee.AutoField and field.name == 'id':
                continue
            column = Column(field)
            field_code.append(column.to_code())
            self.modules.update(column.modules)

        meta = ['class Meta:',
                INDENT + 'table_name = "%s"' % model._meta.table_name]

        if model._meta.schema:
            meta.append(INDENT + 'schema = "%s"' % model._meta.schema)

        if model._meta.indexes:
            meta.append(INDENT + 'indexes = (')
            for index_obj in model._meta.indexes:
                if isinstance(index_obj, peewee.Node):
                    index = NodeDeconstructor().get_code(index_obj)
                else:
                    index = index_obj
                meta.append(INDENT * 2 + repr(index) + ',')
            meta.append(INDENT * 2 + ')')

        if model._meta.constraints:
            meta.append(INDENT + 'constraints = %r' % (
                            get_constraints(model._meta.constraints)))

        if model._meta.primary_key is not None:
            if isinstance(model._meta.primary_key, peewee.CompositeKey):
                names = ', '.join(repr(x) for x in model._meta.primary_key.field_names)
                meta.append(INDENT + 'primary_key = peewee.CompositeKey(%s)' % names)
            elif model._meta.primary_key is False:
                meta.append(INDENT + 'primary_key = False')
        return template.format(
            classname=model.__name__,
            fields=('\n' + INDENT).join(field_code),
            meta=('\n' + INDENT).join(meta)
        )


class Storage:

    def __init__(self, database, models, migrate_table='migratehistory', **kwargs):
        self.database = database
        self.models = models
        self.migrate_table = migrate_table

    @cached_property
    def history_model(self):
        """Initialize and cache MigrationHistory model."""
        class MigrateHistory(peewee.Model):
            name = peewee.CharField()
            migrated = peewee.DateTimeField(default=datetime.utcnow)

            class Meta:
                database = self.database
                table_name = self.migrate_table
                # schema = self.schema

            def __str__(self):
                return self.name

        MigrateHistory.create_table(True)
        return MigrateHistory

    @property
    def todo(self):
        raise NotImplementedError()

    @property
    def done(self):
        return [x.name for x in self.history_model.select().order_by(self.history_model.id)]

    @property
    def undone(self):
        done = set(self.done)
        return [name for name in self.todo if name not in done]

    def set_done(self, name):
        self.history_model.insert({self.history_model.name: name}).execute()

    def set_undone(self, name):
        self.history_model.delete().where(self.history_model.name == name).execute()

    def get_last_step(self):
        return (['zero'] + self.todo)[-1]

    def get_steps(self, name):
        direction = 'backward'
        if name:
            name = self.find_name(name) or name
        else:
            if not self.todo:
                return [], direction
            name = self.todo[-1]
        if name == 'zero':
            steps = ['zero'] + self.done
        elif name not in self.todo:
            raise KeyError(name)
        elif name in self.done:
            steps = self.done[self.done.index(name):]
        else:
            direction = 'forward'
            steps = (['zero'] + self.done)[-1:] + self.undone[:self.undone.index(name) + 1]
        if direction == 'backward':
            steps.reverse()
        return list(zip(steps[:-1], steps[1:])), direction

    def find_name(self, name):
        try:
            prefix = '{:04}_'.format(int(name))
            for todo in self.todo:
                if todo.startswith(prefix):
                    return todo
        except ValueError:
            pass

    def get_name(self, name):
        name = name or datetime.now().strftime('migration_%Y%m%d%H%M')
        return '{:04}_{}'.format(len(self.todo) + 1, name)

    def exec(self, code):
        scope = {'Snapshot': lambda: Snapshot(self.database, self.models)}
        code = compile(code, '<string>', 'exec', dont_inherit=True)
        exec(code, scope)
        allowed_attrs = ('snapshot', 'forward', 'backward', 'migrate_forward', 'migrate_backward')
        return {k: v for k, v in scope.items() if k in allowed_attrs}

    def read(self, name):
        if name == 'zero':
            scope = {'snapshot': Snapshot(self.database, self.models)}
        else:
            return self.exec(self._read(name))
        return scope

    def _read(self, name):
        raise NotImplementedError

    def clear(self):
        self.history_model._schema.drop_all()


class FileStorage(Storage):

    filemask = re.compile(r"[\d]{4}_[^.]+\.py$")

    def __init__(self, *args, migrate_dir='migrations', **kwargs):
        super().__init__(*args, **kwargs)
        self.migrate_dir = migrate_dir

    @property
    def todo(self):
        if not os.path.exists(self.migrate_dir):
            os.makedirs(self.migrate_dir)
        todo = list(sorted(f[:-3] for f in os.listdir(self.migrate_dir) if self.filemask.match(f)))
        return todo

    def _read(self, name):
        with open(os.path.join(self.migrate_dir, name + '.py')) as f:
            return f.read()

    def write(self, name, code):
        with open(os.path.join(self.migrate_dir, name + '.py'), 'w') as f:
            f.write(code)

    def clear(self):
        super().clear()
        for name in self.todo:
            os.remove(os.path.join(self.migrate_dir, name + '.py'))


class MemoryStorage(Storage):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._migrations = {}

    @property
    def todo(self):
        return sorted(self._migrations)

    def _read(self, name):
        return self._migrations[name]

    def write(self, name, code):
        self._migrations[name] = code

    def clear(self):
        super().clear()
        self._migrations = {}


class Router:

    def __init__(self, database, models, storage_class=FileStorage, **storage_kwargs):
        self.database = database
        self.models = models
        self.storage = storage_class(database=database, models=models, **storage_kwargs)

    @property
    def done(self):
        return self.storage.done

    @property
    def undone(self):
        return self.storage.undone

    @property
    def todo(self):
        return self.storage.todo

    def clear(self):
        self.storage.clear()

    def create(self, name=None, tracecode=None, serialize=False, atomic=True):
        """Create a migration."""
        name = self.storage.get_name(name)
        last_step = self.storage.get_last_step()
        last_snap = self.storage.read(last_step)['snapshot']
        compiler = Compiler(self.database, self.models)
        if compiler.snapshot == Compiler(self.database, last_snap).snapshot:
            return []
        alerts = [name]
        if tracecode is not None:
            tracecode['code'] = compiler.module_code

        new_snap = self.storage.exec(compiler.module_code)['snapshot']
        migrator = Migrator(self.database, name, last_snap.get_orm(), new_snap.get_orm(),
                            compute_hints=True, serialize=serialize, atomic=atomic)
        migrator.migrate()

        def add_code(funcname, code):
            code_num_lines = len(compiler.get_code().splitlines())
            for linenum, alert in code.get_alerts():
                alerts.append((linenum + code_num_lines, alert))
            compiler.add_code(code.get_code(funcname))

        if migrator.forward_hints:
            add_code('forward', migrator.forward_hints)

        if migrator.serialized:
            add_code('migrate_forward', migrator.serialized)

        migrator = Migrator(self.database, name, new_snap.get_orm(), last_snap.get_orm(),
                            compute_hints=True, serialize=serialize, atomic=atomic)
        migrator.migrate()

        if migrator.forward_hints:
            add_code('backward', migrator.forward_hints)

        if migrator.serialized:
            add_code('migrate_backward', migrator.serialized)

        self.storage.write(name, compiler.get_code())
        return alerts

    def migrate(self, migration=None, atomic=True):
        """Run migration."""
        try:
            steps, direction = self.storage.get_steps(migration)
        except KeyError:
            raise MigrationError('Unknown migration `%s`.' % migration)
        result_steps = []
        for name_from, name_to in steps:
            step_from = self.storage.read(name_from)
            step_to = self.storage.read(name_to)
            if direction == 'forward':
                step_name = name_to
                migrate_data = step_to.get('forward')
                serialized_migration = step_to.get('migrate_forward')
            else:
                step_name = name_from
                migrate_data = step_from.get('backward')
                serialized_migration = step_from.get('migrate_backward')
            migrator = Migrator(self.database,
                                step_name,
                                step_from['snapshot'].get_orm(),
                                step_to['snapshot'].get_orm(),
                                migrate_data,
                                serialized_migration,
                                atomic=atomic)
            migrator.migrate()
            migrator.direction = direction
            if direction == 'forward':
                migrator.add_operation(self.storage.set_done, args=(step_name,), forced=True)
            else:
                migrator.add_operation(self.storage.set_undone, args=(step_name,), forced=True)
            result_steps.append(migrator)
        return result_steps


class State:

    pk_columns1: list
    pk_columns2: list
    drop_constraints: list
    drop_indexes: list
    add_fields: list
    drop_fields: list
    check_fields: list
    add_not_null: list
    drop_not_null: list
    add_indexes: list
    add_constraints: list

    def __init__(self, **kwargs):
        self.__dict__ = kwargs


HintDescription = namedtuple('HintDescription', ['test', 'exec', 'final'])


def get_operations_for_db(database, **kwargs):
    if isinstance(database, peewee.PostgresqlDatabase):
        return PostgresqlOperations(database, **kwargs)
    elif isinstance(database, peewee.MySQLDatabase):
        return MySQLOperations(database, **kwargs)
    else:
        raise NotImplementedError('Sqlite is not supported')


class Migrator:
    """Provide migrations."""

    forward_hints = None
    backward_hints = None

    def __init__(self, database, name, old_orm, new_orm, run_data_migration=None, run_serialized=None, *,
                 compute_hints=False, serialize=False, atomic=True):
        self.database = database
        self.name = name
        self.old_orm = old_orm
        self.new_orm = new_orm
        self.run_serialized = run_serialized
        self.operations = []
        self.serialize = serialize
        self.serialized = MigrateCode(('op', 'old_orm', 'new_orm'), True)
        if compute_hints:
            self.compute_hints = True
            self.forward_hints = MigrateCode(('old_orm', 'new_orm'))
            self.backward_hints = MigrateCode(('old_orm', 'new_orm'))
        else:
            self.compute_hints = False

        self.op = get_operations_for_db(database, atomic=atomic, data_migration=run_data_migration,
                                        old_orm=old_orm, new_orm=new_orm)

        if serialize:
            self.recorder = OperationSerializer(self.op, self.old_orm, self.new_orm, self.serialized.add_operation)
        else:
            self.recorder = OperationRecorder(self.op, self.add_operation)

    # @classmethod
    # def add_hint(cls, type1, type2, *tests, final=True):
    #     if type1 is None:
    #         def f1(_): return True
    #     else:
    #         def f1(obj): return isinstance(obj.old_field, type1)
    #     if type2 is None:
    #         def f2(_): return True
    #     else:
    #         def f2(obj): return isinstance(obj.new_field, type2)
    #     assert all(callable(x) for x in tests)
    #
    #     def appender(fn):
    #         cls.hints.append(
    #             HintDescription(
    #                 test=(lambda obj: f1(obj) and f2(obj) and all(f(obj) for f in tests)),
    #                 exec=fn,
    #                 final=final
    #             ))
    #         return fn
    #     return appender

    def add_operation(self, obj, *, args=None, kwargs=None, color=None, forced=False):
        if isinstance(obj, list):
            for op in obj:
                self.add_operation(op, color=color, forced=forced)
        elif isinstance(obj, (peewee.Node, peewee.Context)):
            self.operations.append((SQLOP(obj, self.database), color, forced))
        elif callable(obj):
            self.operations.append((PYOP(obj, args, kwargs), color, forced))
        else:
            raise TypeError('Invalid operation.')

    def run(self, fake=False, skip=0, ignore_errors=False):
        for idx, (op, color, forced) in enumerate(self.operations):
            try:
                can_run = forced or (not fake and idx >= skip)
                if can_run:
                    with self.op.transaction():
                        op.run()
                yield op.description, color, not can_run
            except Exception as e:
                if not ignore_errors:
                    raise MigrationError(str(e), op.description)
                else:
                    yield op.description + ": " + str(e), 'ERROR', True
        self.operations = []

    def get_ops(self):
        return list(self.operations)

    def migrate(self):
        if self.run_serialized:
            return self.run_serialized(op=self.recorder, old_orm=self.old_orm, new_orm=self.new_orm)

        models1 = peewee.sort_models(list(self.old_orm))
        models2 = peewee.sort_models(list(self.new_orm))
        models1 = OrderedDict([(m._meta.name, m) for m in models1])
        models2 = OrderedDict([(m._meta.name, m) for m in models2])

        # Add models
        deferred_fields = []
        for name in [m for m in models2 if m not in models1]:
            for field in models2[name]._meta.sorted_fields:
                if isinstance(field, peewee.ForeignKeyField) and field.deferred:
                    deferred_fields.append(field)
            self.recorder.create_table(models2[name])

        if deferred_fields:
            for field in deferred_fields:
                self.recorder.create_foreign_key(field)

        models_to_migrate = [(models1[name], models2[name]) for name in models1 if name in models2]
        if models_to_migrate:
            self._migrate_models(models_to_migrate)
        else:
            self.recorder.run_data_migration()

        # Remove models
        for name in [m for m in reversed(models1) if m not in models2]:
            self.recorder.drop_table(models1[name])

    @staticmethod
    def _is_index_for_foreign_key(index):
        return (isinstance(index, peewee.Index)
                and len(index._expressions) == 1
                and isinstance(index._expressions[0], peewee.ForeignKeyField))

    @staticmethod
    def _get_primary_key_columns(model):
        return tuple(f.column_name for f in model._meta.get_primary_keys())

    def _get_indexes(self, model):
        result = {}
        for index in model._meta.fields_to_index():
            if self._is_index_for_foreign_key(index):
                continue
            ddl = self.op.ctx().sql(index).query()[0]
            result[ddl] = index
        return result

    def _field_type(self, field):
        ctx = self.op.ctx()
        return ctx.sql(field.ddl_datatype(ctx)).query()[0]

    def _get_foreign_key_constraints(self, model):
        result = {}
        for field in model._meta.sorted_fields:
            if isinstance(field, peewee.ForeignKeyField):
                ddl = self.op.ctx().sql(field.foreign_key_constraint()).query()[0]
                result[(ddl, field.unique)] = field
        return result

    def _migrate_models(self, pairs):
        state = {}

        for pair in pairs:
            # init state for each pair
            state[pair] = self._render_migrate_state(*pair)

        for pair in pairs:
            self._prepare_model(state[pair], *pair)

        for pair in pairs:
            self._update_model(state[pair], *pair)

        self.recorder.run_data_migration()

        for pair in pairs:
            self._cleanup_model(state[pair], *pair)

    def _render_migrate_state(self, model1, model2):
        indexes1 = self._get_indexes(model1)
        indexes2 = self._get_indexes(model2)
        constraints1 = self._get_foreign_key_constraints(model1)
        constraints2 = self._get_foreign_key_constraints(model2)
        fields1 = model1._meta.fields
        fields2 = model2._meta.fields

        return State(
            pk_columns1=self._get_primary_key_columns(model1),
            pk_columns2=self._get_primary_key_columns(model2),
            drop_indexes=[indexes1[key] for key in set(indexes1) - set(indexes2)],
            add_indexes=[indexes2[key] for key in set(indexes2) - set(indexes1)],
            drop_constraints=[constraints1[key] for key in set(constraints1) - set(constraints2)],
            add_constraints=[constraints2[key] for key in set(constraints2) - set(constraints1)],
            drop_fields=[fields1[key] for key in set(fields1) - set(fields2)],
            add_fields=[fields2[key] for key in set(fields2) - set(fields1)],
            check_fields=[(fields1[key], fields2[key]) for key in set(fields1).intersection(fields2)],
            add_not_null=[],
            drop_not_null=[]
        )

    def _prepare_model(self, state, model1, _):
        if state.pk_columns1 and state.pk_columns1 != state.pk_columns2:
            self.recorder.drop_primary_key_constraint(model1)

        for field in state.drop_constraints:
            self.recorder.drop_foreign_key_constraint(field)

        for index in state.drop_indexes:
            self.recorder.drop_index(model1, index._name)

    def _update_model(self, state, _, __):
        for field in state.add_fields:
            self.recorder.add_column(field)
            if not field.null:
                state.add_not_null.append(field)
            self.add_data_migration_hints(None, field, False)

        for field in state.drop_fields:
            self.add_data_migration_hints(field, None, False)

        for field1, field2 in state.check_fields:
            if self._field_type(field1) != self._field_type(field2):
                self.recorder.rename_column(field1, "old__" + field1.column_name)
                self.recorder.add_column(field2)
                state.drop_fields.append(field1)
                if not field2.null:
                    state.add_not_null.append(field2)
                self.add_data_migration_hints(field1, field2, True)
                continue

            if field1.column_name != field2.column_name:
                self.recorder.rename_column(field1, field2.column_name)

            if field1.null != field2.null:
                if field2.null:
                    state.drop_not_null.append(field2)
                else:
                    state.add_not_null.append(field2)
                self.add_data_migration_hints(field1, field2, False)

    def _cleanup_model(self, state, _, model2):
        for field in state.drop_fields:
            self.recorder.drop_column(field)

        for field in state.add_not_null:
            self.recorder.add_not_null(field)

        for field in state.drop_not_null:
            self.recorder.drop_not_null(field)

        for index in state.add_indexes:
            self.recorder.add_index(model2, index._name)

        for field in state.add_constraints:
            self.recorder.add_foreign_key_constraint(field)

        # if pk_columns2 and pk_columns2 != pk_columns1:
        #     self.add_primary_key_constraint(model2)

    def add_data_migration_hints(self, field1, field2, type_changed):
        if not self.compute_hints:
            return

        def run_helper(helper_func, old_field, new_field, code):
            kwargs = {
                'postgres': isinstance(self.database, peewee.PostgresqlDatabase),
                'mysql': isinstance(self.database, peewee.MySQLDatabase),
                'old_field': old_field,
                'new_field': new_field,
                'old_model': ('old_' + old_field.model._meta.name) if old_field else None,
                'new_model': new_field.model._meta.name if new_field else None,
            }
            result = [x.format(**kwargs) for x in helper_func(**kwargs)]
            if kwargs['old_model']:
                code.set_var_if_not_exists(kwargs['old_model'], 'old_orm[%r]' % old_field.model._meta.name)
            if kwargs['new_model']:
                code.set_var_if_not_exists(kwargs['new_model'], 'new_orm[%r]' % kwargs['new_model'])
            code.add_operation(result)

        if (field1 is None or field1.null) and (field2 is not None and not field2.null):
            run_helper(set_not_null_helper, None, field2, self.forward_hints)
        if (field2 is None or field2.null) and (field1 is not None and not field1.null):
            run_helper(set_not_null_helper, None, field1, self.backward_hints)

        if type_changed:
            field_check1 = field1.rel_field if isinstance(field1, peewee.ForeignKeyField) else field1
            field_check2 = field2.rel_field if isinstance(field2, peewee.ForeignKeyField) else field2
            for (tp1, tp2), helper in DATA_MIGRATE_HINTS:
                if isinstance(field_check1, tp1) and isinstance(field_check2, tp2):
                    run_helper(helper, field1, field2, self.forward_hints)
                    break
            for (tp1, tp2), helper in DATA_MIGRATE_HINTS:
                if isinstance(field_check2, tp1) and isinstance(field_check1, tp2):
                    run_helper(helper, field2, field1, self.backward_hints)
                    break


class MigrateCode:

    def __init__(self, args, simple=False):
        self.vars = {}
        self.args = args
        self.results = []
        self.alerts = []
        self.simple = simple

    def __bool__(self):
        return bool(self.results)

    def get_code(self, funcname):
        code = 'def %s(%s):\n' % (funcname, ', '.join(self.args))
        lines = []
        for v in self.vars.items():
            lines.append('%s = %s' % v)
        if not self.results:
            if self.simple:
                lines.append('pass')
            else:
                lines.append('return []')
        else:
            if self.simple:
                for line in self.results:
                    lines.append(line)
            else:
                lines.append('return [')
                for line in self.results:
                    lines.append('    ' + line + ',')
                lines.append(']')
        code += textwrap.indent('\n'.join(lines), ' ' * 4) + '\n'
        return code

    def get_alerts(self):
        start_line = 4 + len(self.vars)
        return [(start_line + line, alert) for line, alert in self.alerts]

    def set_var_if_not_exists(self, name, value):
        if name not in self.vars:
            self.vars[name] = value
            # self.lines.append('%s = %s' % (name, value))

    def add_operation(self, result):
        comment, code, *alerts = result
        if alerts:
            self.alerts.extend([(len(self.results), alert) for alert in alerts])
        if comment:
            self.results.append('# ' + comment)
        if code:
            self.results.append(code)


FIELD_DEFAULTS = (
    (peewee.CharField, ''),
    (peewee.IntegerField, 0),
    (peewee.DecimalField, decimal.Decimal),
    (peewee.DateTimeField, datetime.now),
    (peewee.DateField, date.today),
    (peewee.TimeField, time),
    (peewee.BooleanField, False),
)


def escaped_repr(obj):
    return repr(obj).replace('{', '{{').replace('}', '}}')


def set_not_null_helper(new_field, **_):
    default = new_field.default
    if callable(default):
        default = default()
    if default is None:
        for cls, default_value in FIELD_DEFAULTS:
            if isinstance(new_field, cls):
                default = default_value
                if callable(default):
                    default = default()
                break
    if default is None:
        comment = 'Check the field `{new_model}.{new_field.name}` does not contain null values'
        return comment, '', comment
    else:
        return (
            'Apply default value %s to the field {new_model}.{new_field.name}' % escaped_repr(default),
            '{new_model}.update({{{new_model}.{new_field.name}: %s}})'
            '.where({new_model}.{new_field.name}.is_null(True))' % escaped_repr(default)
        )


def charfield_to_charfield_helper(old_field, new_field, **_):
    if old_field.max_length == new_field.max_length:
        return
    return (
        'Convert datatype of the field {new_model}.{new_field.name}: '
        'VARCHAR({old_field.max_length}) -> VARCHAR({new_field.max_length})',
        '{new_model}.update({{{new_model}.{new_field.name}: '
        'fn.SUBSTRING({old_model}.{old_field.name}, 1, {new_field.max_length})}})'
        '.where({old_model}.{old_field.name}.is_null(False))'
    )


def field_to_charfield_helper(postgres=False, **_):
    typecast = 'VARCHAR' if postgres else 'CHAR'
    return (
        'Convert datatype of the field {new_model}.{new_field.name}: '
        '{old_field.field_type} -> VARCHAR({new_field.max_length})',
        '{new_model}.update({{{new_model}.{new_field.name}: '
        '{old_model}.{old_field.name}.cast(%r)}})'
        '.where({old_model}.{old_field.name}.is_null(False))' % typecast,
        'Check the field `{new_model}.{new_field.name}` is converted correctly to string',
    )


def field_to_integer_helper(postgres=False, **_):
    typecast = 'INTEGER' if postgres else 'SIGNED'
    return (
        'Convert datatype of the field {new_model}.{new_field.name}: '
        '{old_field.field_type} -> {new_field.field_type}',
        '{new_model}.update({{{new_model}.{new_field.name}: '
        '{old_model}.{old_field.name}.cast(%r)}})'
        '.where({old_model}.{old_field.name}.is_null(False))' % typecast,
        'Check the field `{new_model}.{new_field.name}` is converted correctly to integer',
    )


def field_to_field_helper(**_):
    return (
        'Don\'t know how to do the conversion correctly, use the naive',
        '{new_model}.update({{{new_model}.{new_field.name}: {old_model}.{old_field.name}}})'
        '.where({old_model}.{old_field.name}.is_null(False))',
        'Check the field `{new_model}.{new_field.name}` is correctly converted',
    )


DATA_MIGRATE_HINTS = [
    ((peewee.CharField, peewee.CharField), charfield_to_charfield_helper),
    ((peewee.Field, peewee.CharField), field_to_charfield_helper),
    ((peewee.Field, peewee.IntegerField), field_to_integer_helper),
    ((peewee.Field, peewee.Field), field_to_field_helper),
]


class SQLOP:

    __slots__ = ('obj', 'database')

    def __init__(self, obj, database):
        self.obj = obj
        self.database = database

    def run(self):
        self.database.execute(self.obj, scope=peewee.SCOPE_VALUES)

    @property
    def description(self):
        obj = self.obj
        if not isinstance(obj, peewee.Context):
            ctx = self.database.get_sql_context(scope=peewee.SCOPE_VALUES)
            obj = ctx.sql(obj)
        query, params = obj.query()
        return 'SQL> %s %s' % (query, params)


class PYOP:

    __slots__ = ('fn', 'args', 'kwargs')

    def __init__(self, fn, args, kwargs):
        self.fn = fn
        self.args = args or ()
        self.kwargs = kwargs or {}

    def run(self):
        self.fn(*self.args, **self.kwargs)

    @property
    def description(self):
        params = []
        if self.args:
            params += ['%r' % a for a in self.args]
        if self.kwargs:
            params += ['%s=%r' % (k, v) for k, v in self.kwargs.items()]
        return 'PY>  %s(%s)' % (self.fn.__name__, ', '.join(params))


class OperationRecorder:
    def __init__(self, obj, recorder):
        self.obj = obj
        self.recorder = recorder

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)
        obj = getattr(self.obj, attr)
        if not callable(obj) or not getattr(obj, "is_operation", False):
            raise AttributeError(attr)

        def recorder(*args, **kwargs):
            result = obj(*args, **kwargs)
            self.recorder(result)
            return result

        return recorder


class OperationSerializer:
    def __init__(self, obj, old_orm, new_orm, recorder):
        self.obj = obj
        self.old_orm = old_orm
        self.new_orm = new_orm
        self.recorder = recorder

    def __find_orm(self, model):
        if model in self.old_orm:
            return "old_orm.%s" % model._meta.name
        return "new_orm.%s" % model._meta.name

    def __serialize(self, obj):
        if isinstance(obj, peewee.Field):
            model = getattr(obj, "original_model", obj.model)
            return self.__find_orm(model) + "." + obj.name
        elif isinstance(obj, type) and issubclass(obj, peewee.Model):
            return self.__find_orm(obj)
        elif isinstance(obj, (int, float, str)):
            return repr(obj)
        assert 0, "Should not be here: %r" % obj

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)
        obj = getattr(self.obj, attr)
        if not callable(obj) or not getattr(obj, "is_operation", False):
            raise AttributeError(attr)

        def recorder(*args, **kwargs):
            s_args = [self.__serialize(arg) for arg in args]
            s_kwargs = ["%s=%s" % (k, self.__serialize(v)) for k, v in kwargs.items()]
            self.recorder(("", "op.%s(%s)" % (obj.__name__, ", ".join(s_args + s_kwargs))))

        return recorder


def operation(fn):
    fn.is_operation = True
    return fn


class Operations:

    def __init__(self, database, atomic, old_orm, new_orm, data_migration=None):
        self._database = database
        self._atomic = atomic
        self._old_orm = old_orm
        self._new_orm = new_orm
        self._data_migration = data_migration

    def transaction(self):
        return self._database.atomic()

    def ctx(self):
        return self._database.get_sql_context(scope=peewee.SCOPE_VALUES)

    def alter_table(self, model):
        return (self.ctx()
                    .literal('ALTER TABLE ')
                    .sql(model))

    @operation
    def run_data_migration(self):
        if self._data_migration:
            return self._data_migration(old_orm=self._old_orm, new_orm=self._new_orm)
        return []

    @operation
    def sql(self, query):
        return self.ctx().literal(query)

    @operation
    def create_table(self, model):
        operations = []
        if self._database.sequences:
            for field in model._meta.sorted_fields:
                if field and field.sequence:
                    operations.append(model._schema._create_sequence(field))
        operations.append(model._schema._create_table(safe=False))
        operations.extend(model._schema._create_indexes(safe=False))
        return operations

    @operation
    def drop_table(self, model):
        return model._schema._drop_table(safe=False, cascade=True)

    @operation
    def add_index(self, model, index):
        for idx in model._meta.fields_to_index():
            if idx._name == index:
                return self._add_index(model, idx)
        raise NameError("Unknown index name %r" % index)

    @operation
    def drop_index(self, model, index):
        for idx in model._meta.fields_to_index():
            if idx._name == index:
                return self._drop_index(model, idx)
        raise NameError("Unknown index name %r" % index)

    @operation
    def add_column(self, field):
        model = field.model
        field = field.clone()
        field.null = True
        field.primary_key = False
        ctx = self.alter_table(model)
        return (ctx.literal(' ADD COLUMN ')
                   .sql(field.ddl(ctx)))

    @operation
    def create_foreign_key(self, field):
        return field.model._schema._create_foreign_key(field)

    @operation
    def drop_column(self, field):
        return (self.alter_table(field.model)
                    .literal(' DROP COLUMN ')
                    .sql(field))

    @operation
    def rename_column(self, field, column_name):
        res = self._rename_column(field.model, field, field.column_name, column_name)
        if field.model in self._old_orm:
            # rename and change model if possible
            name = field.model._meta.name
            if self._new_orm.get(name):
                field.model = self._new_orm[name]
        field.column_name = column_name
        return res

    @operation
    def add_primary_key_constraint(self, model):
        pk_columns = [f.column for f in model._meta.get_primary_keys()]
        ctx = self.alter_table(model).literal(' ADD PRIMARY KEY ')
        return ctx.sql(peewee.EnclosedNodeList(pk_columns))

    @operation
    def drop_primary_key_constraint(self, model):
        return self._drop_primary_key_constraint(model)

    @operation
    def drop_foreign_key_constraint(self, field):
        model = field.model
        index = peewee.ModelIndex(model, (field,), unique=field.unique)
        return [
            self._drop_foreign_key_constraint(model, field),
            self.drop_index(model, index._name)
        ]

    @operation
    def add_foreign_key_constraint(self, field):
        model = field.model
        index = peewee.ModelIndex(model, (field,), unique=field.unique)
        return [
            self.add_index(model, index._name),
            self._add_foreign_key_constraint(model, field)
        ]

    @operation
    def add_not_null(self, field):
        return self._add_not_null(field.model, field)

    @operation
    def drop_not_null(self, field):
        return self._drop_not_null(field.model, field)

    def _add_index(self, model, index):
        return model._schema._create_index(index, safe=False)

    def _drop_index(self, model, index):
        return model._schema._drop_index(index, safe=False)

    def _get_primary_key_field(self, field):
        return field

    def _rename_column(self, model, field, column_name_from, column_name_to):
        raise NotImplementedError

    def _drop_primary_key_constraint(self, model):
        raise NotImplementedError

    def _add_foreign_key_constraint(self, model, field):
        return self.alter_table(model).literal(' ADD ').sql(field.foreign_key_constraint())

    def _drop_foreign_key_constraint(self, model, name):
        raise NotImplementedError

    def _add_not_null(self, model, field):
        raise NotImplementedError

    def _drop_not_null(self, model, field):
        raise NotImplementedError


class LazyQuery(peewee.Node):

    def __init__(self):
        self.ops = []
        self.computed = {}

    def __getattr__(self, attr):
        def tracker(value):
            self.ops.append((attr, value))
            return self
        return tracker

    def __sql__(self, ctx):
        for attr, value in self.ops:
            if callable(value):
                if value not in self.computed:
                    self.computed[value] = value()
                value = self.computed[value]
            ctx = getattr(ctx, attr)(value)
        return ctx


lazy_query = LazyQuery


class PostgresqlOperations(Operations):

    def transaction(self):
        if self._atomic:
            return self._database.atomic()
        return self._pg_autocommit()

    @contextlib.contextmanager
    def _pg_autocommit(self):
        """Gets the bare cursor which has no transaction context."""
        connection = self._database.connection()
        old_isolation_level = connection.isolation_level
        connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        try:
            yield
        finally:
            connection.set_isolation_level(old_isolation_level)

    def _add_index(self, model, index):
        query = super()._add_index(model, index)
        if not self._atomic:
            query._sql[0] += "CONCURRENTLY "
        return query

    def _drop_index(self, model, index):
        query = super()._drop_index(model, index)
        if not self._atomic:
            query._sql[0] += "CONCURRENTLY "
        return query

    def _add_not_null(self, model, field):
        return (self.alter_table(model)
                    .literal(' ALTER COLUMN ')
                    .sql(field)
                    .literal(' SET NOT NULL'))

    def _drop_not_null(self, model, field):
        return (self.alter_table(model)
                    .literal(' ALTER COLUMN ')
                    .sql(field)
                    .literal(' DROP NOT NULL'))

    def _rename_column(self, model, field, column_name_from, column_name_to):
        column_from = peewee.Column(model._meta.table, column_name_from)
        column_to = peewee.Column(model._meta.table, column_name_to)
        return (self.alter_table(model)
                    .literal(' RENAME COLUMN ')
                    .sql(column_from)
                    .literal(' TO ')
                    .sql(column_to))

    def _get_primary_key_constraint_name(self, table_name, schema):
        sql = (
            "SELECT DISTINCT tc.constraint_name "
            "FROM information_schema.table_constraints AS tc "
            "WHERE tc.constraint_type = 'PRIMARY KEY' AND "
            "tc.table_name = %s AND "
            "tc.table_schema = %s"
        )
        cursor = self._database.execute_sql(sql, (table_name, schema))
        result = cursor.fetchall()
        return peewee.Entity(result[0][0] if result else '?')

    def _drop_primary_key_constraint(self, model):
        table_name = model._meta.table_name
        schema = model._meta.schema or 'public'
        return (lazy_query().literal('ALTER TABLE ')
                            .sql(peewee.Entity(table_name))
                            .literal(' DROP CONSTRAINT ')
                            .sql(lambda: self._get_primary_key_constraint_name(table_name, schema)))

    def _add_foreign_key_constraint(self, model, field):
        if self._atomic:
            return super()._add_foreign_key_constraint(model, field)

        table_name = model._meta.table_name
        schema = model._meta.schema or 'public'
        rel_table_name = field.rel_model._meta.table_name
        rel_schema = field.rel_model._meta.schema or 'public'
        column_name = field.column_name
        rel_column_name = field.rel_field.column_name
        return [
            super()._add_foreign_key_constraint(model, field).literal(' NOT VALID'),
            lazy_query().literal('ALTER TABLE ')
                        .sql(peewee.Entity(table_name))
                        .literal(' VALIDATE CONSTRAINT ')
                        .sql(lambda: self._get_foreign_key_constraint_name(
                            table_name, schema, column_name, rel_table_name, rel_schema, rel_column_name
                        ))
        ]

    def _get_foreign_key_constraint_name(self, table_name, schema, column_name,
                                         rel_table_name, rel_schema, rel_column_name):
        sql = (
            "SELECT tc.constraint_name "
            "FROM information_schema.table_constraints AS tc "
            "JOIN information_schema.key_column_usage AS kcu "
            "ON (tc.constraint_name = kcu.constraint_name AND "
            "tc.constraint_schema = kcu.constraint_schema) "
            "JOIN information_schema.constraint_column_usage AS ccu "
            "ON (ccu.constraint_name = tc.constraint_name AND "
            "ccu.constraint_schema = tc.constraint_schema) "
            "WHERE "
            "tc.constraint_type = 'FOREIGN KEY' AND "
            "tc.table_name = %s AND "
            "tc.table_schema = %s AND "
            "ccu.table_name = %s AND "
            "ccu.table_schema = %s AND "
            "kcu.column_name = %s AND "
            "ccu.column_name = %s"
        )
        cursor = self._database.execute_sql(sql, (table_name, schema, rel_table_name, rel_schema,
                                                  column_name, rel_column_name))
        result = cursor.fetchall()
        return peewee.Entity(result[0][0] if result else '?')

    def _drop_foreign_key_constraint(self, model, field):
        table_name = model._meta.table_name
        schema = model._meta.schema or 'public'
        rel_table_name = field.rel_model._meta.table_name
        rel_schema = field.rel_model._meta.schema or 'public'
        column_name = field.column_name
        rel_column_name = field.rel_field.column_name
        return (lazy_query().literal('ALTER TABLE ')
                            .sql(peewee.Entity(table_name))
                            .literal(' DROP CONSTRAINT ')
                            .sql(lambda: self._get_foreign_key_constraint_name(
                                    table_name, schema, column_name, rel_table_name, rel_schema, rel_column_name
                            )))


class MySQLOperations(Operations):

    @operation
    def drop_index(self, model, index):
        return (model._schema._drop_index(index, safe=False)
                     .literal(' ON ')
                     .sql(model))

    def _rename_column(self, model, field, column_name_from, column_name_to):
        column_from = peewee.Column(model._meta.table, column_name_from)
        column_to = peewee.Column(model._meta.table, column_name_to)
        ctx = self.alter_table(model)
        (ctx.literal(' CHANGE ')
            .sql(column_from).literal(' ')
            .sql(column_to).literal(' ')
            .sql(field.ddl_datatype(ctx)))
        if not field.null:
            ctx.literal(' NOT NULL')
        return ctx

    def _add_not_null(self, model, field):
        ctx = self.alter_table(model)
        return (ctx.literal(' MODIFY ')
                   .sql(field.ddl(ctx)))

    _drop_not_null = _add_not_null

    def _drop_primary_key_constraint(self, model):
        pk = model._meta.primary_key
        operations = []
        if isinstance(pk, peewee.AutoField):
            field = pk.clone()
            field.primary_key = False
            field.__class__ = peewee.IntegerField
            ctx = self.alter_table(model)
            operations.append(ctx.literal(' CHANGE ')
                                 .sql(pk).literal(' ')
                                 .sql(field.ddl(ctx)))
        operations.append(self.alter_table(model).literal(' DROP PRIMARY KEY'))
        return operations

    def _get_foreign_key_constraint_name(self, table_name, column_name, rel_table_name, rel_column_name):
        sql = (
            "SELECT constraint_name "
            "FROM information_schema.key_column_usage "
            "WHERE table_name = %s "
            "AND column_name = %s "
            "AND table_schema = DATABASE() "
            "AND referenced_table_name = %s "
            "AND referenced_column_name = %s"
        )
        cursor = self._database.execute_sql(sql, (table_name, column_name, rel_table_name, rel_column_name))
        result = cursor.fetchall()
        return peewee.Entity(result[0][0] if result else '?')

    def _drop_foreign_key_constraint(self, model, field):
        table_name = model._meta.table_name
        column_name = field.column_name
        rel_table_name = field.rel_model._meta.table_name
        rel_column_name = field.rel_field.column_name
        return (lazy_query().literal('ALTER TABLE ')
                            .sql(peewee.Entity(table_name))
                            .literal(' DROP FOREIGN KEY ')
                            .sql(lambda: self._get_foreign_key_constraint_name(
                                        table_name, column_name, rel_table_name, rel_column_name
                            )))

    def _get_primary_key_field(self, field):
        if isinstance(field, peewee.AutoField):
            params = Column(field, complete=True).to_params()
            result = peewee.IntegerField(**params)
            return result
        return field
