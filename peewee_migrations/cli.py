import os
import sys
import types
import json
import importlib
import importlib.util
from traceback import format_exc

import click
import peewee
from . migrator import Router, MigrationError


@click.group()
@click.option('-c', '--config', default='migrations.json')
@click.pass_context
def cli(ctx, config):
    ctx.obj = {
        'config': config
    }
    if '' not in sys.path:
        sys.path[0:0] = ['']


@cli.command()
@click.pass_context
def init(ctx):
    """
    Create new configuration file (default: migrations.json)
    """
    params = ctx.obj
    if os.path.exists(params['config']):
        click.secho('Configuration file %r already exists.' % params['config'], fg='red')
        sys.exit(2)

    init_config(params['config'])


def init_config(filename):
    with open(filename, 'wt') as f:
        data = {
            'prerun': '# some code here',
            'directory': 'migrations',
            'history': 'migratehistory',
            'models': []
        }
        json.dump(data, f, indent=2)

    click.secho('Configuration file %r was created.' % filename, fg='green')


def import_module(path):
    attrs = []
    while 1:
        try:
            module = importlib.import_module(path)
            rest = '.'.join(reversed(attrs))
            return module, rest
        except ModuleNotFoundError as e:
            try:
                path, attr = path.rsplit('.', 1)
            except ValueError:
                raise e
            attrs.append(attr)


def import_string(path, module=None):
    if module is None:
        module, path = import_module(path)
    ret = module
    if path:
        for attr in path.split('.'):
            ret = getattr(ret, attr)
    return ret


def import_models(path, *, seen=None):
    if isinstance(path, str):
        obj = import_string(path)
    else:
        obj = path

    seen = seen or set()

    if obj not in seen and hasattr(obj, '__watch_models__'):
        seen.add(obj)
        watch = obj.__watch_models__
        if callable(watch):
            watch = watch()
        for obj in watch:
            if isinstance(obj, str) and obj.startswith('.'):
                obj = path + obj
            yield from import_models(obj, seen=seen)
        return

    if not isinstance(obj, type) or not issubclass(obj, peewee.Model):
        click.secho('Can\'t load model %s, not a peewee model' % obj, fg='red')
        sys.exit(3)

    yield obj


@cli.command()
@click.pass_context
@click.argument('model')
def add(ctx, model):
    """
    Start watching model
    """
    params = ctx.obj

    if not os.path.exists(params['config']):
        init_config(params['config'])

    with open(params['config'], 'rt') as f:
        data = json.load(f)

    models = data.setdefault('models', [])

    if model in models:
        click.secho(' %r already in the watch list.' % model, fg='red')
        sys.exit(1)

    import_models(model)
    models.append(model)

    with open(params['config'], 'wt') as f:
        json.dump(data, f, indent=2)

    click.secho('Model %r was added to watch list.' % model, fg='green')


@cli.command()
@click.pass_context
@click.argument('model')
def remove(ctx, model):
    """
    Stop watching model
    """
    params = ctx.obj

    if not os.path.exists(params['config']):
        init_config(params['config'])

    with open(params['config'], 'rt') as f:
        data = json.load(f)

    models = data.setdefault('models', [])

    if model not in models:
        click.secho('%r is not in the watch list.' % model, fg='red')
        sys.exit(1)

    models.remove(model)

    with open(params['config'], 'wt') as f:
        json.dump(data, f, indent=2)

    click.secho('%r was removed from the watch list.' % model, fg='green')


def load_conf(ctx):
    filename = ctx.obj['config']
    with open(filename, 'rt') as f:
        conf = json.load(f)

    if conf.get('prerun'):
        if isinstance(conf['prerun'], list):
            prerun = '\n'.join(conf['prerun'])
        else:
            prerun = conf['prerun']
        exec(prerun, {})

    errors = False
    if not conf.get('directory'):
        click.secho('%s: set `directory`' % filename, fg='red')
        errors = True
    if not conf.get('history'):
        click.secho('%s: set `history`' % filename, fg='red')
        errors = True
    if not conf.get('models'):
        click.secho('%s: set `models`' % filename, fg='red')
        errors = True

    if errors:
        sys.exit(3)

    databases = set()
    models = []
    for path in conf['models']:
        for model in import_models(path):
            db = model._meta.database
            if isinstance(db, peewee.Proxy):
                db = db.obj
            if db is None:
                click.secho('Database is not specified for model %s.' % model.__name__, fg='red')
                sys.exit(3)
            databases.add(db)
            models.append(model)

    conf['models'] = models

    if len(databases) != 1:
        click.secho('%r: found multiple database instances, only one is supported.' % filename, fg='red')
        sys.exit(3)

    conf['db'] = databases.pop()

    return conf


def load_router(ctx):
    conf = load_conf(ctx)
    router = Router(database=conf['db'],
                    models=conf['models'],
                    migrate_dir=conf['directory'],
                    migrate_table=conf['history'])
    return router


@cli.command()
@click.option('--showsql', is_flag=True, help='Only show sql')
@click.pass_context
def createtables(ctx, showsql):
    """Create tables"""
    conf = load_conf(ctx)
    if showsql:
        def fake_execute_sql(sql, params, *args, **kwargs):
            click.secho('SQL> %s %s' % (sql, params), fg='yellow')
        conf['db'].execute_sql = fake_execute_sql
    conf['db'].create_tables(conf['models'], safe=False)


@cli.command()
@click.pass_context
def droptables(ctx):
    """Drop tables"""
    conf = load_conf(ctx)
    conf['db'].drop_tables(conf['models'], safe=True)


@cli.command()
@click.option('--traceback', is_flag=True, help='Show traceback')
@click.option('--tracecode', is_flag=True, help='Show generated code')
@click.option('--serialize', is_flag=True, help='Show generated code')
@click.pass_context
def watch(ctx, traceback, tracecode, serialize):
    """Watch model changes and create migration"""

    router = load_router(ctx)
    if tracecode:
        context = {}
    else:
        context = None
    try:
        result = router.create(tracecode=context, serialize=serialize)
    except Exception as e:
        click.secho('Migration error: ' + str(e), fg='red')
        if tracecode and context.get('code'):
            lines = context['code'].splitlines()
            code = [(click.style('% 5d ' % num, fg='yellow') + click.style(line, fg='white'))
                    for num, line in enumerate(lines, 1)]
            click.echo('\n'.join(code))
        if traceback:
            click.secho(format_exc(), fg='yellow', bold=True)
        return
    if not result:
        click.secho('No changes found.', fg='green')
    else:
        click.secho('Migration `%s` has been created.' % result[0], fg='cyan')
        for line, text in result[1:]:
            text = ''.join([click.style(j, 'magenta' if i % 2 else 'yellow')
                            for i, j in enumerate(text.split('`'))])
            click.echo(click.style('Line %d: ' % line, 'yellow') + text)


@cli.command(name='list')
@click.pass_context
def list_(ctx):
    """List migrations"""

    router = load_router(ctx)
    for name in router.done:
        click.secho('[X] ' + name, fg='green')
    for name in router.undone:
        click.secho('[ ] ' + name, fg='yellow')


@cli.command()
@click.argument('to', required=False, default=None)
@click.option('-a', '--autocommit', default=False, is_flag=True, help='run without a transaction')
@click.option('-s', '--skip', default=0, type=int, help='skip first N operations')
@click.option('-e', '--explicit_schema', default=None, type=str, help='explicit schema name')
@click.pass_context
def show(ctx, to, autocommit, skip, explicit_schema):
    """Show migrations"""

    router = load_router(ctx)
    atomic = not autocommit
    try:
        steps = router.migrate(to, atomic=atomic, explicit_schema=explicit_schema)
    except MigrationError as e:
        click.secho('Migration error: ' + str(e), fg='red')
        sys.exit(1)

    if not steps:
        click.secho('There is nothing to migrate', fg='yellow')
        return

    for step in steps:
        if step.run_serialized:
            serialized = ' (serialized)'
        else:
            serialized = ''
        if step.direction == 'forward':
            click.secho('[ ] ' + step.name + serialized + ':', fg='yellow')
        else:
            click.secho('[X] ' + step.name + serialized + ':', fg='green')
        for idx, (op, color, _) in enumerate(step.get_ops()):
            if idx >= skip:
                prefix = "[*]"
            else:
                prefix = "[ ]"
            if color == 'ALERT':
                click.secho('  %s %s' % (prefix, op.description), fg='magenta')
            else:
                click.secho('  %s %s' % (prefix, op.description), fg='cyan')
        click.echo('')


@cli.command()
@click.argument('to', required=False, default=None)
@click.option('-f', '--fake', default=False, is_flag=True, help='fake migration')
@click.option('-s', '--skip', default=0, type=int, help='skip first N operations')
@click.option('-a', '--autocommit', default=False, is_flag=True, help='run without a transaction')
@click.option('-i', '--ignore-errors', default=False, is_flag=True, help='ignore errors and try to continue')
@click.option('-e', '--explicit_schema', default=None, type=str, help='explicit schema name')
@click.option('--traceback', is_flag=True, help='Show traceback')
@click.pass_context
def migrate(ctx, to, fake, skip, autocommit, traceback, ignore_errors, explicit_schema):
    """Run migrations"""

    router = load_router(ctx)

    atomic = not autocommit
    try:
        steps = router.migrate(to, atomic=atomic, explicit_schema=explicit_schema)
    except MigrationError as e:
        click.secho('Migration error: ' + str(e), fg='red')
        if traceback:
            click.secho(format_exc(), fg='yellow', bold=True)
        sys.exit(1)

    if not steps:
        click.secho('There is nothing to migrate', fg='yellow')
        return

    for step in steps:
        if step.run_serialized:
            serialized = ' (serialized)'
        else:
            serialized = ''
        if step.direction == 'forward':
            click.secho('[X] ' + step.name + serialized, fg='green')
        else:
            click.secho('[ ] ' + step.name + serialized, fg='yellow')
        try:
            for descr, color, skipped in step.run(fake=fake, skip=skip, ignore_errors=ignore_errors):
                if skipped:
                    prefix = "[ ]"
                else:
                    prefix = "[*]"
                if color == 'ALERT':
                    click.secho('  %s %s' % (prefix, descr), fg='magenta')
                elif color == 'ERROR':
                    click.secho('  %s %s' % (prefix, descr), fg='red')
                else:
                    click.secho('  %s %s' % (prefix, descr), fg='cyan')
            skip = 0
            click.echo('')
        except MigrationError as e:
            click.secho('  [!] %s' % e.args[1], fg='red')
            click.echo('')
            click.secho('Migration error: ' + e.args[0], fg='red')
            if traceback:
                click.secho(format_exc(), fg='yellow', bold=True)
            raise

def run():
    cli()


if __name__ == '__main__':
    cli()
