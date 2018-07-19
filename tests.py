import os
import datetime
import unittest
from peewee import *
from apistar_peewee.migrator import Migrator, Column, Router, MemoryStorage, MigrationError
from playhouse.reflection import Introspector as BaseIntrospector
from playhouse.reflection import Metadata as BaseMetadata
from playhouse.reflection import PostgresqlMetadata as BasePostgresqlMetadata
from playhouse.reflection import MySQLMetadata as BaseMySQLMetadata


def db_loader(engine, name, db_class=None, **params):
    if db_class is None:
        engine_aliases = {
            SqliteDatabase: ['sqlite', 'sqlite3'],
            MySQLDatabase: ['mysql'],
            PostgresqlDatabase: ['postgres', 'postgresql'],
        }
        engine_map = dict((alias, db) for db, aliases in engine_aliases.items()
                          for alias in aliases)
        if engine.lower() not in engine_map:
            raise Exception('Unsupported engine: %s.' % engine)
        db_class = engine_map[engine.lower()]
    if issubclass(db_class, SqliteDatabase) and not name.endswith('.db'):
        name = '%s.db' % name if name != ':memory:' else name
    return db_class(name, **params)


BACKEND = os.environ.get('BACKEND') or 'sqlite'
PRINT_DEBUG = os.environ.get('PRINT_DEBUG')

IS_SQLITE = BACKEND in ('sqlite', 'sqlite3')
IS_MYSQL = BACKEND == 'mysql'
IS_POSTGRESQL = BACKEND in ('postgres', 'postgresql')


def new_connection():
    return db_loader(BACKEND, 'apistar_peewee_test')


db = new_connection()


class Metadata(BaseMetadata):

    def get_column_type_modifiers(self, table, schema):
        query = """
            SELECT column_name, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = %%s AND table_schema = %s""" % schema
        cursor = self.database.execute_sql(query, (table, ))
        result = []
        for name, max_length in cursor.fetchall():
            if max_length is not None:
                result.append((name, {'max_length': max_length}))
        return result

    def get_columns(self, table, schema=None):
        columns = super().get_columns(table, schema)
        def make_field_class(field_class, extra_kwargs):
            def maker(*args, **kwargs):
                kwargs.update(extra_kwargs)
                return field_class(*args, **kwargs)
            return maker
        for name, modifiers in self.get_column_type_modifiers(table, schema):
            columns[name].field_class = make_field_class(columns[name].field_class, modifiers)
        return columns


class PostgresqlMetadata(BasePostgresqlMetadata, Metadata):

    def get_column_type_modifiers(self, table, schema):
        return super().get_column_type_modifiers(table, "'%s'" % (schema or 'public'))


class MySQLMetadata(BaseMySQLMetadata, Metadata):

    def get_column_type_modifiers(self, table, schema):
        return super().get_column_type_modifiers(table, 'DATABASE()')


class Introspector(BaseIntrospector):

    @classmethod
    def from_database(cls, database, schema=None):
        if isinstance(database, PostgresqlDatabase):
            metadata = PostgresqlMetadata(database)
        elif isinstance(database, MySQLDatabase):
            metadata = MySQLMetadata(database)
        else:
            raise NotImplementedError('Not yet implemented')
        # else:
        #     metadata = SqliteMetadata(database)
        return cls(metadata, schema=schema)


class TestModel(Model):
    class Meta:
        database = db


class MigrationTestCase(unittest.TestCase):

    database = db

    def setUp(self):
        if not self.database.is_closed():
            self.database.close()
        self.database.connect()
        self.introspector = Introspector.from_database(self.database)
        self.router = Router(self.database, storage_class=MemoryStorage)
        self.cleanup = []
        super().setUp()

    def tearDown(self):
        super().tearDown()
        self.database.drop_tables(self.cleanup, safe=True)
        self.router.clear()
        self.database.close()

    def add_cleanup(self, cls):
        self.cleanup.append(cls)
        return cls

    def makemigrations(self, models):
        name = self.router.create(models=models)[0]
        if PRINT_DEBUG:
            print(self.router.storage._read(name))

    def migrate(self):
        steps = self.router.migrate()
        if PRINT_DEBUG:
            for step in steps:
                print(step.name)
                for op, color in step.get_ops():
                    print(op.description)
                print()
        [step.run() for step in steps]

    def get_models(self):
        return self.introspector.generate_models()

    def assertModelsEqual(self, model1, model2):
        self.assertEqual(model1._meta.table_name, model2._meta.table_name)
        indexes1 = sorted([(tuple(names), unique) for names, unique in model1._meta.indexes])
        indexes2 = sorted([(tuple(names), unique) for names, unique in model2._meta.indexes])
        self.assertEqual(indexes1, indexes2)
        exclude = ['backref', 'default', 'on_delete', 'on_update']
        fields1 = {k: (type(v), Column(v).to_params(exclude)) for k, v in model1._meta.fields.items()}
        fields2 = {k: (type(v), Column(v).to_params(exclude)) for k, v in model2._meta.fields.items()}
        if PRINT_DEBUG:
            print('assertModelsEqual:')
            print(fields1)
            print(fields2)
        self.assertDictEqual(fields1, fields2)


class MigrationTests(MigrationTestCase):

    def test_add_model(self):
        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=100, unique=True)

        self.makemigrations([A])
        self.migrate()

        self.assertModelsEqual(A, self.get_models()['a'])

    def test_drop_models(self):
        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=100, unique=True)

        @self.add_cleanup
        class B(TestModel):
            col = ForeignKeyField(A)

        self.makemigrations([A, B])
        self.makemigrations([])
        self.migrate()

        self.assertFalse('a' in self.get_models())
        self.assertFalse('b' in self.get_models())

    def test_add_index(self):
        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=100)

        self.makemigrations([A])

        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=100, unique=True)

        self.makemigrations([A])
        self.migrate()

        self.assertModelsEqual(A, self.get_models()['a'])

    def test_drop_index(self):
        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=100, index=True)

        self.makemigrations([A])

        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=100)

        self.makemigrations([A])
        self.migrate()

        self.assertModelsEqual(A, self.get_models()['a'])

    def test_add_field(self):
        @self.add_cleanup
        class A(TestModel):
            col1 = CharField(max_length=100)

        self.makemigrations([A])
        self.migrate()

        A.create(col1='aaa')
        A.create(col1='bbb')

        @self.add_cleanup
        class A(TestModel):
            col1 = CharField(max_length=100)
            col2 = CharField(max_length=100, default='')

        self.makemigrations([A])
        self.migrate()

        self.assertModelsEqual(A, self.get_models()['a'])

    def test_drop_field(self):
        @self.add_cleanup
        class A(TestModel):
            col1 = CharField(max_length=100)
            col2 = CharField(max_length=100)

        self.makemigrations([A])

        @self.add_cleanup
        class A(TestModel):
            col1 = CharField(max_length=100)

        self.makemigrations([A])
        self.migrate()

        self.assertModelsEqual(A, self.get_models()['a'])

    def test_alter_field(self):
        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=100)

        self.makemigrations([A])

        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=50)

        self.makemigrations([A])
        self.migrate()

        self.assertModelsEqual(A, self.get_models()['a'])

    def test_alter_with_data(self):
        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=5)

        self.makemigrations([A])
        self.migrate()

        A.create(col='abcde')

        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=10)

        self.makemigrations([A])
        self.migrate()

        data = A.select().first()
        self.assertEqual(data.col, 'abcde')
        self.assertModelsEqual(A, self.get_models()['a'])

    def test_alter_with_data2(self):
        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=10)

        self.makemigrations([A])
        self.migrate()

        A.create(col='abcdefghij')

        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=5)

        self.makemigrations([A])
        self.migrate()

        data = A.select().first()

        self.assertEqual(data.col, 'abcde')
        self.assertModelsEqual(A, self.get_models()['a'])

    def test_composite_index(self):
        @self.add_cleanup
        class A(TestModel):
            col1 = CharField(max_length=100, unique=True)
            col2 = CharField(max_length=100)
            class Meta:
                indexes = [
                    (('col1', 'col2'), False)
                ]

        self.makemigrations([A])

        @self.add_cleanup
        class A(TestModel):
            col1 = CharField(max_length=100, index=True)
            col2 = CharField(max_length=100)
            col3 = CharField(max_length=100, unique=True, null=True)
            class Meta:
                indexes = [
                    (('col2', 'col3'), True)
                ]

        self.makemigrations([A])
        self.migrate()

        self.assertModelsEqual(A, self.get_models()['a'])

    def test_add_drop_field_add_drop_index(self):
        @self.add_cleanup
        class A(TestModel):
            col1 = CharField(max_length=100)
            col2 = CharField(max_length=200, index=True)

        self.makemigrations([A])

        @self.add_cleanup
        class A(TestModel):
            col1 = CharField(max_length=100, unique=True)
            col3 = CharField(max_length=200)

        self.makemigrations([A])
        self.migrate()

        self.assertModelsEqual(A, self.get_models()['a'])

    def test_not_null_and_default(self):
        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=100, null=True)

        self.makemigrations([A])
        self.migrate()

        A.create(col=None)
        A.create(col='val0')

        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=100, default='val1')

        self.makemigrations([A])
        self.migrate()

        data = list(A.select().order_by(A.col))

        self.assertEqual(data[0].col, 'val0')
        self.assertEqual(data[1].col, 'val1')
        self.assertModelsEqual(A, self.get_models()['a'])

    def test_add_foreign_key(self):
        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=10)

        @self.add_cleanup
        class B(TestModel):
            col = IntegerField()

        self.makemigrations([A, B])

        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=10)

        @self.add_cleanup
        class B(TestModel):
            col = IntegerField()
            col2 = ForeignKeyField(A)

        self.makemigrations([A, B])
        self.migrate()

        self.assertModelsEqual(B, self.get_models()['b'])

    def test_add_foreign_key_constraint(self):
        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=10)

        @self.add_cleanup
        class B(TestModel):
            col = IntegerField()

        self.makemigrations([A, B])
        self.migrate()

        B.create(col=A.create(col='aaa').id)

        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=10)

        @self.add_cleanup
        class B(TestModel):
            col = ForeignKeyField(A)

        self.makemigrations([A, B])
        self.migrate()

        self.assertModelsEqual(A, self.get_models()['a'])
        self.assertModelsEqual(B, self.get_models()['b'])

    def test_drop_foreign_key_constraint(self):
        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=10)

        @self.add_cleanup
        class B(TestModel):
            col = ForeignKeyField(A, unique=True)

        self.makemigrations([A, B])
        self.migrate()

        B.create(col=A.create(col='aaa'))

        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=10)

        @self.add_cleanup
        class B(TestModel):
            col = IntegerField(unique=True)

        self.makemigrations([A, B])
        self.migrate()

        self.assertModelsEqual(A, self.get_models()['a'])
        self.assertModelsEqual(B, self.get_models()['b'])

    def test_alter_foreign_key(self):
        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=10)

        @self.add_cleanup
        class B(TestModel):
            col = ForeignKeyField(A)

        self.makemigrations([A, B])
        self.migrate()

        B.create(col=A.create(col='aaa'))

        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=10)

        @self.add_cleanup
        class B(TestModel):
            col = ForeignKeyField(A, on_delete='CASCADE')

        self.makemigrations([A, B])
        self.migrate()

        self.assertModelsEqual(A, self.get_models()['a'])
        self.assertModelsEqual(B, self.get_models()['b'])

    def test_alter_foreign_key_index(self):
        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=10)

        @self.add_cleanup
        class B(TestModel):
            col = ForeignKeyField(A)

        self.makemigrations([A, B])

        @self.add_cleanup
        class A(TestModel):
            col = CharField(max_length=10)

        @self.add_cleanup
        class B(TestModel):
            col = ForeignKeyField(A, unique=True)

        self.makemigrations([A, B])
        self.migrate()

        self.assertModelsEqual(A, self.get_models()['a'])
        self.assertModelsEqual(B, self.get_models()['b'])

    # def test_drop_primary_key(self):

    # def test_add_drop_primary_key(self):
    #     snapshot = self.get_snapshot()

    #     @self.add_cleanup
    #     class A(TestModel):
    #         col = CharField(max_length=10, primary_key=True)

    #     self.run_migration('test_add_primary_key1')

    #     A.create(col='1')
    #     A.create(col='2')

    #     snapshot = self.get_snapshot()

    #     @self.add_cleanup
    #     class A(TestModel):
    #         col = CharField(max_length=10)
    #         class Meta:
    #             primary_key = False

    #     self.run_migration('test_add_primary_key2')

    #     snapshot = self.get_snapshot()

    #     @self.add_cleanup
    #     class Model1(TestModel):
    #         test1 = CharField(max_length=10)

    #     migrator = self.get_migrator('test_add_primary_key2')

    #     migrator.migrate()

    #     self.run_migrator(migrator)

    #     self.assertModelsEqual(Model1, self.get_models()['model1'])

    # def test_add_compisite_primary_key(self):
    #     snapshot = self.get_snapshot()

    #     @self.add_cleanup
    #     class Model1(TestModel):
    #         test1 = CharField(max_length=10)
    #         test2 = CharField(max_length=10)
    #         class Meta:
    #             primary_key = False

    #     snapshot = self.get_snapshot()

    #     @self.add_cleanup
    #     class Model1(TestModel):
    #         test1 = CharField(max_length=10)
    #         test2 = CharField(max_length=10)
    #         class Meta:
    #             primary_key = CompositeKey('test1', 'test2')

    #     # Just test it works, reflection adds index for primary key
    #     self.run_migration('test_add_compisite_primary_key')

    # def test_change_primary_key_to_default(self):
    #     snapshot = self.get_snapshot()

    #     @self.add_cleanup
    #     class Model1(TestModel):
    #         test1 = CharField(max_length=10, primary_key=True)

    #     self.run_migration('test_change_primary_key_to_default1')

    #     Model1.create(test1='1')
    #     Model1.create(test1='2')

    #     snapshot = self.get_snapshot()

    #     @self.add_cleanup
    #     class Model1(TestModel):
    #         test1 = CharField(max_length=10)

    #     migrator = self.get_migrator('test_change_primary_key_to_default2')

    #     @migrator.before_set_not_null('id')
    #     def set_id(model, field):
    #         cast_to = 'integer' if IS_POSTGRESQL else 'signed'
    #         model.update({field: model.test1.cast(cast_to)}).execute()

    #     migrator.migrate()

    #     self.run_migrator(migrator)

    #     self.assertModelsEqual(Model1, self.get_models()['model1'])

    # def test_change_primary_key_to_custom(self):
    #     snapshot = self.get_snapshot()

    #     @self.add_cleanup
    #     class Model1(TestModel):
    #         test1 = CharField(max_length=10)

    #     snapshot = self.get_snapshot()

    #     @self.add_cleanup
    #     class Model1(TestModel):
    #         test1 = CharField(max_length=10, primary_key=True)

    #     self.run_migration('test_change_primary_key_to_custom')
    #     self.assertModelsEqual(Model1, self.get_models()['model1'])

if __name__ == '__main__':
    unittest.main()
