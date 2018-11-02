import os
import sys
import click
import importlib
import peewee
import json
from traceback import format_exc
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


def import_model(path):
    module, attr = path.rsplit('.', 1)
    return getattr(importlib.import_module(module), attr)


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
        click.secho('Model %r already in watch list.' % model, fg='red')
        sys.exit(1)

    try:
        obj = import_model(model)
    except Exception:
        click.secho('Can`t add model %r to watch list.' % model, fg='red')
        sys.exit(1)

    if not isinstance(obj, type) or not issubclass(obj, peewee.Model):
        click.secho('Can`t add model %r to watch list, not a model' % model, fg='red')
        sys.exit(1)

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
        click.secho('Model %r is not in the watch list.' % model, fg='red')
        sys.exit(1)

    models.remove(model)

    with open(params['config'], 'wt') as f:
        json.dump(data, f, indent=2)

    click.secho('Model %r was removed from the watch list.' % model, fg='green')


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
    for m in conf['models']:
        module, attr = m.rsplit('.', 1)
        m = getattr(importlib.import_module(module), attr)
        databases.add(m._meta.database)
        models.append(m)
    conf['models'] = models

    if len(databases) != 1:
        click.secho('%r: ambiguous database' % filename, fg='red')
        sys.exit(3)

    database = databases.pop()
    if isinstance(database, peewee.Proxy):
        database = database.obj

    conf['db'] = database

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
@click.pass_context
def watch(ctx, traceback, tracecode):
    """Watch model changes and create migration"""

    router = load_router(ctx)
    if tracecode:
        context = {}
    else:
        context = None
    try:
        result = router.create(tracecode=context)
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
@click.option('-t', '--to', default=None, help='target migration')
@click.pass_context
def show(ctx, to):
    """Show migrations"""

    router = load_router(ctx)

    try:
        steps = router.migrate(to)
    except MigrationError as e:
        click.secho('Migration error: ' + str(e), fg='red')
        sys.exit(1)

    if not steps:
        click.secho('There is nothing to migrate', fg='yellow')
        return

    for step in steps:
        if step.direction == 'forward':
            click.secho('[ ] ' + step.name + ':', fg='yellow')
        else:
            click.secho('[X] ' + step.name + ':', fg='green')
        for op, color, _ in step.get_ops():
            if color == 'ALERT':
                click.secho('  %s' % op.description, fg='magenta')
            else:
                click.secho('  %s' % op.description, fg='cyan')
        click.echo('')


@cli.command()
@click.option('-f', '--fake', default=False, is_flag=True, help='fake migration')
@click.argument('to', required=False, default=None)
@click.pass_context
def migrate(ctx, to, fake):
    """Run migrations"""

    router = load_router(ctx)

    try:
        steps = router.migrate(to)
    except MigrationError as e:
        click.secho('Migration error: ' + str(e), fg='red')
        sys.exit(1)

    if not steps:
        click.secho('There is nothing to migrate', fg='yellow')
        return

    for step in steps:
        step.run(fake=fake)
        if step.direction == 'forward':
            click.secho('[X] ' + step.name, fg='green')
        else:
            click.secho('[ ] ' + step.name, fg='yellow')


def run():
    cli()


if __name__ == '__main__':
    cli()
