from django.contrib import admin

# Register your models here.
from django.contrib import admin
from models import Category
from models import Pattern

admin.site.register(Category)
admin.site.register(Pattern)


