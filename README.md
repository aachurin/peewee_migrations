# Peewee Migrations

A simple and flexible migration manager for Peewee ORM.

* **Version:** 0.3.17
* **Status:** Development/Alpha
* **Author:** Churin Andrey

# Requirements

* python >= 3.5
* latest peewee


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
