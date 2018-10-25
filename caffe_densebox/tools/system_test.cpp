#include<cstdio>

void test()
{
	int len=sizeof(int)*8;
	printf("sizeof(int)=%d\n",len);

	len=sizeof(int *)*8;
	printf("sizeof(int*)=%d\n",len);

#ifdef _MSC_VER
	printf("_MSC_VER is defined\n");
#endif


#ifdef __GNUC__
	printf("__GNUC__ is defined\n");
#endif

#ifdef __INTEL__
	printf("__INTEL__  is defined\n");
#endif

#ifdef __i386__
	printf("__i386__  is defined\n");
#endif

#ifdef __x86_64__
	printf("__x86_64__  is defined\n");
#endif

#ifdef _WIN32
	printf("_WIN32 is defined\n");
#endif

#ifdef _WIN64
	printf("_WIN64 is defined\n");
#endif


#ifdef __linux__
	printf("__linux__ is defined\n");
#endif

#ifdef __LP64__
	printf("__LP64__ is defined\n");
#endif


#ifdef __amd64
	printf("__amd64 is defined\n");
#endif

#ifdef __APPLE__
	printf("__APPLE__ is defined\n");
#endif

#ifdef __DARWIN__
	printf("__DARWIN__ is defined\n");
#endif

}

int main(int argc, char* argv[])
{
	test();
	return 0;
}
