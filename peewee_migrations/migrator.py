import os
import re
import peewee
import textwrap
import decimal
from datetime import datetime, date, time
from collections import OrderedDict, namedtuple
from playhouse.postgres_ext import ArrayField


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
        param_str = ', '.join('%s=%r' % (k, v) for k, v in sorted(params.items()))
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


@deconstructor(ArrayField)
def arrayfield_deconstructor(field, params, **_):
    params['dimensions'] = field.dimensions
    params['convert_values'] = field.convert_values
    params['field_class'] = field._ArrayField__field.__class__


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

    def append(self, model):
        model._meta.database = self.database
        self.items.append(model)
        self.mapping[model._meta.name] = model
        return model

    def __repr__(self):
        return repr(self.items)

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
        allowed_attrs = ('snapshot', 'forward', 'backward')
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

    def create(self, name=None, tracecode=None):
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
        migrator = Migrator(self.database, name, last_snap, new_snap, compute_hints=True)
        migrator.migrate()
        if migrator.forward_hints:
            code_num_lines = len(compiler.get_code().splitlines())
            for linenum, alert in migrator.forward_hints.get_alerts():
                alerts.append((linenum + code_num_lines, alert))
            compiler.add_code(str(migrator.forward_hints))
        if migrator.backward_hints:
            code_num_lines = len(compiler.get_code().splitlines())
            for linenum, alert in migrator.backward_hints.get_alerts():
                alerts.append((linenum + code_num_lines, alert))
            compiler.add_code(str(migrator.backward_hints))
        self.storage.write(name, compiler.get_code())
        return alerts

    def migrate(self, migration=None):
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
            else:
                step_name = name_from
                migrate_data = step_from.get('backward')
            migrator = Migrator(self.database,
                                step_name,
                                step_from['snapshot'],
                                step_to['snapshot'],
                                migrate_data)
            migrator.migrate()
            migrator.direction = direction
            if direction == 'forward':
                migrator.add_op(self.storage.set_done, args=(step_name,), forced=True)
            else:
                migrator.add_op(self.storage.set_undone, args=(step_name,), forced=True)
            result_steps.append(migrator)
        return result_steps


def model_field_args(fn):
    def wrapper(self, *args):
        if isinstance(args[0], peewee.Field):
            field, *args = args
            model = field.model
        else:
            model, field, *args = args
            field.model = model
            if getattr(field, 'column_name', None):
                field.name = getattr(field, 'name', field.column_name)
            else:
                field.column_name = field.name
        return fn(self, model, field, *args)
    return wrapper


class State:

    def __init__(self, **kwargs):
        self.__dict__ = kwargs


HintDescription = namedtuple('HintDescription', ['test', 'exec', 'final'])


class Migrator:
    """Provide migrations."""

    forward_hints = ''
    backward_hints = ''

    def __init__(self, database, name, old_orm, new_orm, run_data_migration=None,
                 compute_hints=False):
        self.database = database
        self.name = name
        self.old_orm = old_orm
        self.new_orm = new_orm
        self.run_data_migration = run_data_migration
        self.operations = []
        if compute_hints:
            self.compute_hints = True
            self.forward_hints = MigrateCode('forward', ('old_orm', 'new_orm'))
            self.backward_hints = MigrateCode('backward', ('old_orm', 'new_orm'))
        else:
            self.compute_hints = False
        self.op = OperationProxy(Operations.from_database(database), self.add_op)

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

    def add_op(self, obj, *, args=None, kwargs=None, color=None, forced=False):
        if isinstance(obj, list):
            for op in obj:
                self.add_op(op, color=color, forced=forced)
        elif isinstance(obj, (peewee.Node, peewee.Context)):
            self.operations.append((SQLOP(obj, self.database), color, forced))
        elif callable(obj):
            self.operations.append((PYOP(obj, args, kwargs), color, forced))
        else:
            raise TypeError('Invalid operation.')

    def run(self, fake=False):
        with self.database.transaction():
            for op, _, forced in self.operations:
                try:
                    if forced or not fake:
                        op.run()
                except Exception as e:
                    raise Exception(str(e), op.description)
        self.operations = []

    def get_ops(self):
        return list(self.operations)

    def migrate(self):
        models1 = peewee.sort_models(self.old_orm)
        models2 = peewee.sort_models(self.new_orm)
        models1 = OrderedDict([(m._meta.name, m) for m in models1])
        models2 = OrderedDict([(m._meta.name, m) for m in models2])

        # Add models
        deferred_fields = []
        for name in [m for m in models2 if m not in models1]:
            for field in models2[name]._meta.sorted_fields:
                if isinstance(field, peewee.ForeignKeyField) and field.deferred:
                    deferred_fields.append(field)
            self.op.create_table(models2[name])

        if deferred_fields:
            for field in deferred_fields:
                self.op.create_foreign_key(field)

        models_to_migrate = [(models1[name], models2[name]) for name in models1 if name in models2]
        if models_to_migrate:
            self._migrate_models(models_to_migrate)

        # Remove models
        for name in [m for m in reversed(models1) if m not in models2]:
            self.op.drop_table(models1[name])

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

        if self.run_data_migration:
            ops = list(self.run_data_migration(new_orm=self.new_orm, old_orm=self.old_orm))
            self.add_op(ops, color='ALERT')

        for pair in pairs:
            self._cleanup_model(state[pair], *pair)

    def _render_migrate_state(self, model1, model2):
        indexes1 = self._get_indexes(model1)
        indexes2 = self._get_indexes(model2)
        constaints1 = self._get_foreign_key_constraints(model1)
        constaints2 = self._get_foreign_key_constraints(model2)
        fields1 = model1._meta.fields
        fields2 = model2._meta.fields

        return State(
            pk_columns1=self._get_primary_key_columns(model1),
            pk_columns2=self._get_primary_key_columns(model2),
            drop_indexes=[indexes1[key] for key in set(indexes1) - set(indexes2)],
            add_indexes=[indexes2[key] for key in set(indexes2) - set(indexes1)],
            drop_constraints=[constaints1[key] for key in set(constaints1) - set(constaints2)],
            add_constraints=[constaints2[key] for key in set(constaints2) - set(constaints1)],
            drop_fields=[fields1[key] for key in set(fields1) - set(fields2)],
            add_fields=[fields2[key] for key in set(fields2) - set(fields1)],
            check_fields=[(fields1[key], fields2[key]) for key in set(fields1).intersection(fields2)],
            add_not_null=[],
            drop_not_null=[]
        )

    def _prepare_model(self, state, model1, _):
        if state.pk_columns1 and state.pk_columns1 != state.pk_columns2:
            self.op.drop_primary_key_constraint(model1)

        for field in state.drop_constraints:
            self.op.drop_foreign_key_constraint(field)

        for index in state.drop_indexes:
            self.op.drop_index(model1, index)

    def _update_model(self, state, _, __):
        for field in state.add_fields:
            # if field is model2._meta.primary_key:
            #     field = self._get_primary_key_field(field)
            self.op.add_column(field)
            if not field.null:
                state.add_not_null.append(field)
            self.add_data_migration_hints(None, field, False)

        for field in state.drop_fields:
            self.add_data_migration_hints(field, None, False)

        for field1, field2 in state.check_fields:
            # XXX: hack
            field1.model = field2.model
            if field1.column_name != field2.column_name:
                self.op.rename_column(field1, field2.column_name)
                field1.column_name = field2.column_name

            if self._field_type(field1) != self._field_type(field2):
                old_column_name = 'old__' + field1.column_name
                self.op.rename_column(field1, old_column_name)
                self.op.add_column(field2)
                field1.column_name = old_column_name
                state.drop_fields.append(field1)
                if not field2.null:
                    state.add_not_null.append(field2)
                self.add_data_migration_hints(field1, field2, True)
            elif field1.null != field2.null:
                if field2.null:
                    state.drop_not_null.append(field2)
                else:
                    state.add_not_null.append(field2)
                self.add_data_migration_hints(field1, field2, False)

    def _cleanup_model(self, state, _, model2):
        for field in state.drop_fields:
            self.op.drop_column(field)

        for field in state.add_not_null:
            self.op.add_not_null(field)

        for field in state.drop_not_null:
            self.op.drop_not_null(field)

        for index in state.add_indexes:
            self.op.add_index(model2, index)

        for field in state.add_constraints:
            self.op.add_foreign_key_constraint(field)

        # if pk_columns2 and pk_columns2 != pk_columns1:
        #     self.add_primary_key_constraint(model2)

    def add_data_migration_hints(self, field1, field2, type_changed):
        if not self.compute_hints:
            return

        def run_helper(helper, old_field, new_field, code):
            kwargs = {
                'postgres': isinstance(self.database, peewee.PostgresqlDatabase),
                'mysql': isinstance(self.database, peewee.MySQLDatabase),
                'old_field': old_field,
                'new_field': new_field,
                'old_model': ('old_' + old_field.model._meta.name) if old_field else None,
                'new_model': new_field.model._meta.name if new_field else None,
            }
            result = [x.format(**kwargs) for x in helper(**kwargs)]
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

    def __init__(self, funcname, args):
        self.vars = {}
        self.funcname = funcname
        self.args = args
        self.results = []
        self.alerts = []

    def __bool__(self):
        return bool(self.results)

    def __str__(self):
        code = 'def %s(%s):\n' % (self.funcname, ', '.join(self.args))
        lines = []
        for v in self.vars.items():
            lines.append('%s = %s' % v)
        if not self.results:
            lines.append('return []')
        else:
            lines.append('return [')
            for line in self.results:
                lines.append('    ' + line)
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
            self.results.append(code + ',')


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


class OperationProxy:

    def __init__(self, obj, appender):
        self.obj = obj
        self.appender = appender

    def __getattr__(self, attr):
        obj = getattr(self.obj, attr)
        if hasattr(obj, '__func__') and getattr(obj.__func__, 'is_operation', False):
            def wrapper(*args, **kwargs):
                result = obj(*args, **kwargs)
                self.appender(result)
                return result
            return wrapper
        return obj


def operation(fn):
    fn.is_operation = True
    return fn


class Operations:

    def __init__(self, database):
        self.database = database

    @classmethod
    def from_database(cls, database):
        if isinstance(database, peewee.PostgresqlDatabase):
            return PostgresqlOperations(database)
        elif isinstance(database, peewee.MySQLDatabase):
            return MySQLOperations(database)
        else:
            raise NotImplementedError('Sqlite is not supported')
            # return SqliteOperations(database)

    def ctx(self):
        return self.database.get_sql_context(scope=peewee.SCOPE_VALUES)

    def _alter_table(self, model: peewee.Model):
        return (self.ctx()
                    .literal('ALTER TABLE ')
                    .sql(model))

    @operation
    def create_table(self, model: peewee.Model):
        operations = []
        if self.database.sequences:
            for field in model._meta.sorted_fields:
                if field and field.sequence:
                    operations.append(model._schema._create_sequence(field))
        operations.append(model._schema._create_table(safe=False))
        operations.extend(model._schema._create_indexes(safe=False))
        return operations

    @operation
    def drop_table(self, model: peewee.Model):
        return model._schema._drop_table(safe=False, cascade=True)

    @operation
    def add_index(self, model: peewee.Model, index: peewee.Index):
        return model._schema._create_index(index, safe=False)

    @operation
    def drop_index(self, model: peewee.Model, index: peewee.Index):
        return model._schema._drop_index(index, safe=False)

    @operation
    @model_field_args
    def add_column(self, model: peewee.Model, field: peewee.Field):
        field = field.clone()
        field.null = True
        field.primary_key = False
        ctx = self._alter_table(model)
        return (ctx.literal(' ADD COLUMN ')
                   .sql(field.ddl(ctx)))

    @operation
    @model_field_args
    def create_foreign_key(self, model: peewee.Model, field: peewee.Field):
        return model._schema._create_foreign_key(field)

    @operation
    @model_field_args
    def drop_column(self, model: peewee.Model, field: peewee.Field):
        return (self._alter_table(model)
                    .literal(' DROP COLUMN ')
                    .sql(field))

    @operation
    @model_field_args
    def rename_column(self, model: peewee.Model, field: peewee.Field, column_name: str):
        return self._rename_column(model, field, field.column_name, column_name)

    @operation
    def add_primary_key_constraint(self, model: peewee.Model):
        pk_columns = [f.column for f in model._meta.get_primary_keys()]
        ctx = self._alter_table(model).literal(' ADD PRIMARY KEY ')
        return ctx.sql(peewee.EnclosedNodeList(pk_columns))

    @operation
    def drop_primary_key_constraint(self, model: peewee.Model):
        return self._drop_primary_key_constraint(model)

    @operation
    @model_field_args
    def drop_foreign_key_constraint(self, model: peewee.Model, field: peewee.Field):
        index = peewee.ModelIndex(model, (field,), unique=field.unique)
        return [
            self._drop_foreign_key_constraint(model, field),
            self.drop_index(model, index)
        ]

    @operation
    @model_field_args
    def add_foreign_key_constraint(self, model: peewee.Model, field: peewee.Field):
        index = peewee.ModelIndex(model, (field,), unique=field.unique)
        return [
            self.add_index(model, index),
            self._add_foreign_key_constraint(model, field)
        ]

    @operation
    @model_field_args
    def add_not_null(self, model: peewee.Model, field: peewee.Field):
        return [self._add_not_null(model, field)]

    @operation
    @model_field_args
    def drop_not_null(self, model: peewee.Model, field: peewee.Field):
        return self._drop_not_null(model, field)

    def _get_primary_key_field(self, field):
        return field

    def _rename_column(self, model, field, column_name_from, column_name_to):
        raise NotImplementedError

    def _drop_primary_key_constraint(self, model):
        raise NotImplementedError

    def _add_foreign_key_constraint(self, model, field):
        return self._alter_table(model).literal(' ADD ').sql(field.foreign_key_constraint())

    def _drop_foreign_key_constraint(self, model, name):
        raise NotImplementedError

    def _add_not_null(self, model, field):
        raise NotImplementedError

    def _drop_not_null(self, model, field):
        raise NotImplementedError


class LazyQuery(peewee.Node):

    def __init__(self):
        self.ops = []

    def __getattr__(self, attr):
        def tracker(value):
            self.ops.append((attr, value))
            return self
        return tracker

    def __sql__(self, ctx):
        for attr, value in self.ops:
            if callable(value):
                value = value()
            getattr(ctx, attr)(value)
        return ctx


class PostgresqlOperations(Operations):

    def _add_not_null(self, model, field):
        return (self._alter_table(model)
                    .literal(' ALTER COLUMN ')
                    .sql(field)
                    .literal(' SET NOT NULL'))

    def _drop_not_null(self, model, field):
        return (self._alter_table(model)
                    .literal(' ALTER COLUMN ')
                    .sql(field)
                    .literal(' DROP NOT NULL'))

    def _rename_column(self, model, field, column_name_from, column_name_to):
        column_from = peewee.Column(model._meta.table, column_name_from)
        column_to = peewee.Column(model._meta.table, column_name_to)
        return (self._alter_table(model)
                    .literal(' RENAME COLUMN ')
                    .sql(column_from)
                    .literal(' TO ')
                    .sql(column_to))

    def _drop_primary_key_constraint(self, model):
        params = (model._meta.table_name,
                  model._meta.schema or 'public')

        def get_name():
            sql = """
                SELECT DISTINCT tc.constraint_name
                FROM information_schema.table_constraints AS tc
                WHERE tc.constraint_type = 'PRIMARY KEY' AND
                      tc.table_name = %s AND
                      tc.table_schema = %s
                """
            cursor = self.database.execute_sql(sql, params)
            result = cursor.fetchall()
            return peewee.Entity(result[0][0] if result else '<PRIMARY KEY>')

        return (LazyQuery().literal('ALTER TABLE ')
                           .sql(model._meta.table)
                           .literal(' DROP CONSTRAINT ')
                           .sql(get_name))

    def _drop_foreign_key_constraint(self, model, field):
        params = (model._meta.table_name,
                  model._meta.schema or 'public',
                  field.rel_model._meta.table_name,
                  field.rel_model._meta.schema or 'public',
                  field.column_name,
                  field.rel_field.column_name)

        def get_name():
            sql = """
                SELECT tc.constraint_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON (tc.constraint_name = kcu.constraint_name AND
                        tc.constraint_schema = kcu.constraint_schema)
                JOIN information_schema.constraint_column_usage AS ccu
                    ON (ccu.constraint_name = tc.constraint_name AND
                        ccu.constraint_schema = tc.constraint_schema)
                WHERE
                    tc.constraint_type = 'FOREIGN KEY' AND
                    tc.table_name = %s AND
                    tc.table_schema = %s AND
                    ccu.table_name = %s AND
                    ccu.table_schema = %s AND
                    kcu.column_name = %s AND
                    ccu.column_name = %s
            """
            cursor = self.database.execute_sql(sql, params)
            result = cursor.fetchall()
            return peewee.Entity(result[0][0] if result else '<FOREIGN KEY>')

        return (LazyQuery().literal('ALTER TABLE ')
                           .sql(model._meta.table)
                           .literal(' DROP CONSTRAINT ')
                           .sql(get_name))


class MySQLOperations(Operations):

    @operation
    def drop_index(self, model: peewee.Model, index: peewee.Index):
        return (model._schema._drop_index(index, safe=False)
                     .literal(' ON ')
                     .sql(model))

    def _rename_column(self, model, field, column_name_from, column_name_to):
        column_from = peewee.Column(model._meta.table, column_name_from)
        column_to = peewee.Column(model._meta.table, column_name_to)
        ctx = self._alter_table(model)
        (ctx.literal(' CHANGE ')
            .sql(column_from).literal(' ')
            .sql(column_to).literal(' ')
            .sql(field.ddl_datatype(ctx)))
        if not field.null:
            ctx.literal(' NOT NULL')
        return ctx

    def _add_not_null(self, model, field):
        ctx = self._alter_table(model)
        return (ctx.literal(' MODIFY ')
                   .sql(field.ddl(ctx)))

    _drop_not_null = _add_not_null

    def _drop_primary_key_constraint(self, model: peewee.Model):
        pk = model._meta.primary_key
        operations = []
        if isinstance(pk, peewee.AutoField):
            field = pk.clone()
            field.primary_key = False
            field.__class__ = peewee.IntegerField
            ctx = self._alter_table(model)
            operations.append(ctx.literal(' CHANGE ')
                                 .sql(pk).literal(' ')
                                 .sql(field.ddl(ctx)))
        operations.append(self._alter_table(model).literal(' DROP PRIMARY KEY'))
        return operations

    def _drop_foreign_key_constraint(self, model, field):
        params = (model._meta.table_name,
                  field.column_name,
                  field.rel_model._meta.table_name,
                  field.rel_field.column_name)

        def get_name():
            sql = """
                SELECT constraint_name
                FROM information_schema.key_column_usage
                WHERE table_name = %s
                    AND column_name = %s
                    AND table_schema = DATABASE()
                    AND referenced_table_name = %s
                    AND referenced_column_name = %s
            """
            cursor = self.database.execute_sql(sql, params)
            result = cursor.fetchall()
            return peewee.Entity(result[0][0] if result else '<FOREIGN KEY>')
        return (LazyQuery().literal('ALTER TABLE ')
                           .sql(model._meta.table)
                           .literal(' DROP FOREIGN KEY ')
                           .sql(get_name))

    def _get_primary_key_field(self, field):
        if isinstance(field, peewee.AutoField):
            params = Column(field, complete=True).to_params()
            result = peewee.IntegerField(**params)
            return result
        return field
