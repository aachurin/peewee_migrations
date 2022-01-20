# Peewee Migrations

A simple and flexible migration manager for Peewee ORM.

* **Version:** 0.3.29
* **Status:** Development/Alpha
* **Author:** Churin Andrey

# Requirements

* python >= 3.5
* latest peewee

## Note
SQLite is not supported.


## Quick start

This package can be installed using pip:

```bash
$ pip install peewee-migrations
```

Run `pem init` in the project root.

```bash
$ pem init
Configuration file 'migrations.json' was created.
```

Suppose we have `Foo` model in `models.py`

```python
class Foo(db.Model):
    bar = peewee.CharField(max_length=50)
    baz = peewee.IntegerField()
    quux = peewee.IntegerField()
```

Add this model to the watch list and create migration.

```bash
$ pem add models.Foo
Model 'models.Foo' was added to the watch list.
$ pem watch
Migration `0001_migration_201807191008` has been created.
```

Now you can list available migrations:

```bash
$ pem list
[ ] 0001_migration_201807191008
```

Or view SQL that will be executed during migration:

```bash
$ pem show
[ ] 0001_migration_201807191008:
  SQL> CREATE TABLE "foo" ("id" SERIAL NOT NULL PRIMARY KEY, "bar" VARCHAR(50) NOT NULL, "baz" INTEGER NOT NULL, "quux" INTEGER NOT NULL) []
  PY>  set_done('0001_migration_201807191008')
```

Use `migrate` to run migrations:

```bash
$ pem migrate
[X] 0001_migration_201807191008
```

Change model `Foo`

```python
class Foo(db.Model):
    bar = peewee.CharField(max_length=20)
    baz = peewee.IntegerField()
    quux = peewee.IntegerField()
    xyzzy = peewee.IntegerField()
```

and run `watch` to create new migration:

```bash
$ pem watch
Migration `0002_migration_201807191036` has been created.
$ pem show
[ ] 0002_migration_201807191036:
  SQL> ALTER TABLE "foo" ADD COLUMN "xyzzy" INTEGER []
  SQL> ALTER TABLE "foo" RENAME COLUMN "bar" TO "old__bar" []
  SQL> ALTER TABLE "foo" ADD COLUMN "bar" VARCHAR(20) []
  SQL> UPDATE "foo" SET "xyzzy" = %s WHERE ("xyzzy" IS %s) [0, None]
  SQL> UPDATE "foo" SET "bar" = SUBSTRING("old__bar", %s, %s) WHERE ("old__bar" IS NOT %s) [1, 20, None]
  SQL> ALTER TABLE "foo" DROP COLUMN "old__bar" []
  SQL> ALTER TABLE "foo" ALTER COLUMN "xyzzy" SET NOT NULL []
  SQL> ALTER TABLE "foo" ALTER COLUMN "bar" SET NOT NULL []
  PY>  set_done('0002_migration_201807191036')
```

It's possible to create "serialized" migrations, run `pem watch --serialize`. In this case, explicit migration functions will be additionally created.

```bash
$ pem watch --serialize
Migration `0001_migration_202112161523` has been created.
```

Additional code will be generated

```python
...

def migrate_forward(op, old_orm, new_orm):
    op.create_table(new_orm.foo)


def migrate_backward(op, old_orm, new_orm):
    op.drop_table(old_orm.foo)
```

And after changing the model

```bash
$ pem watch --serialize
Migration `0002_migration_202112161527` has been created.
```

```python
...

def migrate_forward(op, old_orm, new_orm):
    op.add_column(new_orm.foo.xyzzy)
    op.rename_column(old_orm.foo.bar, 'old__bar')
    op.add_column(new_orm.foo.bar)
    op.run_data_migration()
    op.drop_column(old_orm.foo.bar)
    op.add_not_null(new_orm.foo.xyzzy)
    op.add_not_null(new_orm.foo.bar)


...

def migrate_backward(op, old_orm, new_orm):
    op.rename_column(old_orm.foo.bar, 'old__bar')
    op.add_column(new_orm.foo.bar)
    op.run_data_migration()
    op.drop_column(old_orm.foo.xyzzy)
    op.drop_column(old_orm.foo.bar)
    op.add_not_null(new_orm.foo.bar)

```

Serialized migrations are performed only according to the operations specified in the migrate functions.

To run migrations without a transaction, use `pem migrate --autocommit`. To view a list of operations that will be performed in this mode, use `pem show --autocommit` (some operations may differ).

For more information on using the commands see --help.

## migrations.json
```
{
  "prerun": "some code here",  // some code to run before executing any command
  "directory": "migrations",   // folder to hold migrations
  "history": "migratehistory", // table to hold migration history
  "models": [                  // list of models to watch
    "module1.Model1",
    "module2.Model2"
  ]
}
```
